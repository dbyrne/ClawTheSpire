param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [int]$TimeoutSec = 5,
    [int]$PollAttempts = 60,
    [int]$PollDelayMs = 200
)

$ErrorActionPreference = "Stop"

function Invoke-ApiJson {
    param(
        [string]$Method,
        [string]$Path,
        $Body = $null
    )

    $uri = $BaseUrl.TrimEnd("/") + $Path

    try {
        if ($null -eq $Body) {
            $response = Invoke-WebRequest -Method $Method -Uri $uri -UseBasicParsing -TimeoutSec $TimeoutSec
        }
        else {
            $response = Invoke-WebRequest -Method $Method -Uri $uri -UseBasicParsing -TimeoutSec $TimeoutSec -ContentType "application/json" -Body ($Body | ConvertTo-Json -Depth 8 -Compress)
        }

        return $response.Content | ConvertFrom-Json
    }
    catch {
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
            return $_.ErrorDetails.Message | ConvertFrom-Json
        }

        if ($_.Exception.Response -and $_.Exception.Response.GetResponseStream()) {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $content = $reader.ReadToEnd()
            if ($content) {
                return $content | ConvertFrom-Json
            }
        }

        throw
    }
}

function Get-State {
    return (Invoke-ApiJson -Method "GET" -Path "/state").data
}

function Invoke-Action {
    param(
        [hashtable]$Payload
    )

    return Invoke-ApiJson -Method "POST" -Path "/action" -Body $Payload
}

function Wait-ForState {
    param(
        [string]$Description,
        [scriptblock]$Condition
    )

    for ($attempt = 0; $attempt -lt $PollAttempts; $attempt++) {
        $state = Get-State
        if (& $Condition $state) {
            return $state
        }

        Start-Sleep -Milliseconds $PollDelayMs
    }

    throw "Timed out waiting for state: $Description"
}

function Assert-ActionAvailable {
    param(
        $State,
        [string]$ActionName
    )

    if (-not (@($State.available_actions) -contains $ActionName)) {
        throw "Expected action '$ActionName' to be available, but state was: $($State | ConvertTo-Json -Depth 6 -Compress)"
    }
}

function Invoke-DebugCommand {
    param(
        [string]$Command
    )

    $response = Invoke-Action -Payload @{
        action = "run_console_command"
        command = $Command
    }

    if (-not $response.ok) {
        throw "Debug command '$Command' failed: $($response | ConvertTo-Json -Depth 6 -Compress)"
    }

    return $response
}

[void](Invoke-ApiJson -Method "GET" -Path "/health")

$state = Get-State

if ($state.screen -eq "CARD_SELECTION") {
    throw "test-target-index-contract.ps1 expects a stable starting screen, but current screen is CARD_SELECTION."
}

if ($state.screen -eq "MAIN_MENU") {
    Assert-ActionAvailable -State $state -ActionName "continue_run"
    $continueResponse = Invoke-Action -Payload @{ action = "continue_run" }
    if (-not $continueResponse.ok) {
        throw "continue_run failed: $($continueResponse | ConvertTo-Json -Depth 6 -Compress)"
    }

    $state = Wait-ForState -Description "leave MAIN_MENU" -Condition {
        param($CurrentState)
        $CurrentState.screen -ne "MAIN_MENU"
    }
}

while ($state.screen -eq "REWARD") {
    Assert-ActionAvailable -State $state -ActionName "collect_rewards_and_proceed"
    $rewardResponse = Invoke-Action -Payload @{ action = "collect_rewards_and_proceed" }
    if (-not $rewardResponse.ok) {
        throw "collect_rewards_and_proceed failed: $($rewardResponse | ConvertTo-Json -Depth 6 -Compress)"
    }

    $state = Wait-ForState -Description "leave REWARD" -Condition {
        param($CurrentState)
        $CurrentState.screen -ne "REWARD"
    }
}

if (-not $state.in_combat) {
    Invoke-DebugCommand -Command "room Monster" | Out-Null
    $state = Wait-ForState -Description "enter COMBAT" -Condition {
        param($CurrentState)
        $CurrentState.in_combat -and $CurrentState.screen -eq "COMBAT"
    }
}

$emptyPotionSlots = @($state.run.potions | Where-Object { -not $_.occupied })
if ($emptyPotionSlots.Count -eq 0) {
    $discardablePotion = $state.run.potions | Where-Object { $_.occupied -and $_.can_discard } | Select-Object -First 1
    if ($null -eq $discardablePotion) {
        throw "Expected at least one discardable potion slot before injecting BLOCK_POTION."
    }

    $discardResponse = Invoke-Action -Payload @{
        action = "discard_potion"
        option_index = [int]$discardablePotion.index
    }

    if (-not $discardResponse.ok) {
        throw "discard_potion failed while preparing BLOCK_POTION injection: $($discardResponse | ConvertTo-Json -Depth 6 -Compress)"
    }

    $state = Wait-ForState -Description "free potion slot" -Condition {
        param($CurrentState)
        @($CurrentState.run.potions | Where-Object { -not $_.occupied }).Count -gt 0
    }
}

Invoke-DebugCommand -Command "card BELIEVE_IN_YOU hand" | Out-Null
Invoke-DebugCommand -Command "potion BLOCK_POTION" | Out-Null

$state = Wait-ForState -Description "BELIEVE_IN_YOU and BLOCK_POTION injection" -Condition {
    param($CurrentState)

    if ($CurrentState.screen -ne "COMBAT" -or -not $CurrentState.in_combat) {
        return $false
    }

    $hasAnyAllyCard = $null -ne ($CurrentState.combat.hand | Where-Object { $_.card_id -eq "BELIEVE_IN_YOU" } | Select-Object -First 1)
    $hasBlockPotion = $null -ne ($CurrentState.run.potions | Where-Object { $_.occupied -and $_.potion_id -eq "BLOCK_POTION" } | Select-Object -First 1)
    return $hasAnyAllyCard -and $hasBlockPotion
}

$card = $state.combat.hand | Where-Object { $_.card_id -eq "BELIEVE_IN_YOU" } | Select-Object -First 1
if ($null -eq $card) {
    throw "Failed to inject BELIEVE_IN_YOU into the current hand."
}

if ($card.target_type -ne "AnyAlly") {
    throw "Expected BELIEVE_IN_YOU target_type=AnyAlly, but received: $($card | ConvertTo-Json -Depth 6 -Compress)"
}

if (-not [bool]$card.requires_target -or $card.target_index_space -ne "players") {
    throw "Expected BELIEVE_IN_YOU to require combat.players[] targeting, but received: $($card | ConvertTo-Json -Depth 6 -Compress)"
}

$cardTargetIndices = @($card.valid_target_indices | Where-Object { $null -ne $_ } | ForEach-Object { [int]$_ })
if ($cardTargetIndices.Count -ne 0) {
    throw "Expected BELIEVE_IN_YOU to expose no valid_target_indices in singleplayer combat, but received: $($card | ConvertTo-Json -Depth 6 -Compress)"
}

if ([bool]$card.playable -or $card.unplayable_reason -ne "no_living_allies") {
    throw "Expected BELIEVE_IN_YOU to be unplayable with no_living_allies in singleplayer combat, but received: $($card | ConvertTo-Json -Depth 6 -Compress)"
}

$blockPotion = $state.run.potions | Where-Object { $_.occupied -and $_.potion_id -eq "BLOCK_POTION" } | Select-Object -First 1
if ($null -eq $blockPotion) {
    throw "Failed to inject BLOCK_POTION into the current run state."
}

if ($blockPotion.target_type -ne "AnyPlayer") {
    throw "Expected BLOCK_POTION target_type=AnyPlayer, but received: $($blockPotion | ConvertTo-Json -Depth 6 -Compress)"
}

$blockPotionTargetIndices = @($blockPotion.valid_target_indices | Where-Object { $null -ne $_ } | ForEach-Object { [int]$_ })
if ([bool]$blockPotion.requires_target -or -not [string]::IsNullOrWhiteSpace([string]$blockPotion.target_index_space) -or $blockPotionTargetIndices.Count -ne 0) {
    throw "Expected BLOCK_POTION to stay self-targeted in singleplayer combat, but received: $($blockPotion | ConvertTo-Json -Depth 6 -Compress)"
}

$blockBefore = [int]$state.combat.player.block
$usePotionResponse = Invoke-Action -Payload @{
    action = "use_potion"
    option_index = [int]$blockPotion.index
}

if (-not $usePotionResponse.ok) {
    throw "use_potion failed for BLOCK_POTION: $($usePotionResponse | ConvertTo-Json -Depth 6 -Compress)"
}

if ($usePotionResponse.data.status -ne "completed" -or -not [bool]$usePotionResponse.data.stable) {
    throw "Expected BLOCK_POTION to complete immediately without target_index, but received: $($usePotionResponse | ConvertTo-Json -Depth 6 -Compress)"
}

$finalState = $usePotionResponse.data.state
if ([int]$finalState.combat.player.block -le $blockBefore) {
    throw "Expected BLOCK_POTION to increase player block without target_index, but final state was: $($finalState | ConvertTo-Json -Depth 6 -Compress)"
}

[pscustomobject]@{
    screen = $finalState.screen
    any_ally_card = [pscustomobject]@{
        card_id = $card.card_id
        requires_target = [bool]$card.requires_target
        target_index_space = $card.target_index_space
        valid_target_count = $cardTargetIndices.Count
        playable = [bool]$card.playable
        unplayable_reason = $card.unplayable_reason
    }
    any_player_potion = [pscustomobject]@{
        potion_id = $blockPotion.potion_id
        requires_target = [bool]$blockPotion.requires_target
        target_index_space = $blockPotion.target_index_space
        final_block = [int]$finalState.combat.player.block
    }
} | ConvertTo-Json -Depth 6
