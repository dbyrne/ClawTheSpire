using System.Net;
using System.Threading;
using MegaCrit.Sts2.Core.Logging;

namespace STS2AIAgent.Server;

public sealed class HttpServer
{
    private const string DefaultHost = "127.0.0.1";
    private const int DefaultPort = 8080;
    private const string LogPrefix = "[STS2AIAgent.HttpServer]";
    private const int StartRetryCount = 20;
    private static readonly TimeSpan StartRetryDelay = TimeSpan.FromMilliseconds(250);

    private static readonly Lazy<HttpServer> LazyInstance = new(() => new HttpServer());

    private readonly object _gate = new();
    private HttpListener? _listener;
    private CancellationTokenSource? _cts;
    private Task? _listenLoopTask;

    public static HttpServer Instance => LazyInstance.Value;

    private HttpServer()
    {
    }

    public void Start()
    {
        lock (_gate)
        {
            if (_listener != null)
            {
                Log.Info($"{LogPrefix} Already started");
                return;
            }

            var prefix = ResolvePrefix();
            _listener = StartListenerWithRetry(prefix);

            _cts = new CancellationTokenSource();
            _listenLoopTask = Task.Run(() => ListenLoopAsync(_listener, _cts.Token));
            Log.Info($"{LogPrefix} Listening on {prefix}");
        }
    }

    public void Stop()
    {
        HttpListener? listener;
        CancellationTokenSource? cts;
        Task? listenLoopTask;

        lock (_gate)
        {
            if (_listener == null && _cts == null && _listenLoopTask == null)
            {
                return;
            }

            listener = _listener;
            cts = _cts;
            listenLoopTask = _listenLoopTask;
            _listener = null;
            _cts = null;
            _listenLoopTask = null;
        }

        try
        {
            cts?.Cancel();
        }
        catch (Exception ex)
        {
            Log.Warn($"{LogPrefix} Failed to cancel listener token: {ex}");
        }

        try
        {
            if (listener?.IsListening == true)
            {
                listener.Stop();
            }
        }
        catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException)
        {
            Log.Info($"{LogPrefix} Listener stop completed with shutdown exception: {ex.Message}");
        }

        try
        {
            listener?.Close();
        }
        catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException)
        {
            Log.Info($"{LogPrefix} Listener close completed with shutdown exception: {ex.Message}");
        }

        try
        {
            listenLoopTask?.Wait(TimeSpan.FromSeconds(2));
        }
        catch (AggregateException ex) when (ex.InnerExceptions.All(inner => inner is OperationCanceledException or HttpListenerException or ObjectDisposedException))
        {
            Log.Info($"{LogPrefix} Listener loop stopped during shutdown.");
        }
        finally
        {
            cts?.Dispose();
        }

        Log.Info($"{LogPrefix} Stopped");
    }

    private static async Task ListenLoopAsync(HttpListener listener, CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            HttpListenerContext? context = null;

            try
            {
                context = await listener.GetContextAsync();
                _ = Task.Run(() => Router.HandleAsync(context, cancellationToken), cancellationToken);
            }
            catch (HttpListenerException) when (cancellationToken.IsCancellationRequested || !listener.IsListening)
            {
                break;
            }
            catch (ObjectDisposedException) when (cancellationToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                Log.Error($"{LogPrefix} Listener loop failed: {ex}");

                if (context != null)
                {
                    await Router.WriteErrorAsync(context.Response, 500, "listener_error", "HTTP listener failed.");
                }
            }
        }
    }

    private static HttpListener StartListenerWithRetry(string prefix)
    {
        for (var attempt = 1; ; attempt++)
        {
            var listener = new HttpListener();
            listener.Prefixes.Add(prefix);

            try
            {
                listener.Start();
                return listener;
            }
            catch (HttpListenerException ex) when (IsPrefixConflict(ex) && attempt < StartRetryCount)
            {
                listener.Close();
                Log.Warn($"{LogPrefix} Prefix still busy, retrying start ({attempt}/{StartRetryCount - 1})...");
                Thread.Sleep(StartRetryDelay);
            }
        }
    }

    private static string ResolvePrefix()
    {
        var rawPort = Environment.GetEnvironmentVariable("STS2_API_PORT");
        if (!string.IsNullOrWhiteSpace(rawPort) &&
            int.TryParse(rawPort.Trim(), out var port) &&
            port is > 0 and <= 65535)
        {
            return $"http://{DefaultHost}:{port}/";
        }

        return $"http://{DefaultHost}:{DefaultPort}/";
    }

    private static bool IsPrefixConflict(HttpListenerException ex)
    {
        return ex.ErrorCode == 183 ||
            ex.NativeErrorCode == 183 ||
            ex.Message.Contains("conflicts with an existing registration", StringComparison.OrdinalIgnoreCase);
    }
}
