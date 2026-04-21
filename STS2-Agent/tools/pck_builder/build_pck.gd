extends SceneTree

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() != 2:
		push_error("Usage: -- <source_manifest_path> <output_pck_path>")
		quit(1)
		return

	var source_manifest := args[0]
	var output_pck := args[1]
	var packer := PCKPacker.new()

	var start_error := packer.pck_start(output_pck)
	if start_error != OK:
		push_error("Failed to start PCK packer: %s" % start_error)
		quit(1)
		return

	var add_error := packer.add_file("res://mod_manifest.json", source_manifest)
	if add_error != OK:
		push_error("Failed to add mod_manifest.json: %s" % add_error)
		quit(1)
		return

	var flush_error := packer.flush()
	if flush_error != OK:
		push_error("Failed to finalize PCK: %s" % flush_error)
		quit(1)
		return

	print("Packed %s" % output_pck)
	quit(0)
