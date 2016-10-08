package = "torch-nn-ext"
version = "scm-1"

source = {
	url = "git://github.com/anoidgit/Torch-nn-ext",
	tag = "master"
}

description = {
	summary = "some extention to torch and nn package",
	detailed = [[
A library to based on torch.
	]],
	homepage = "https://github.com/anoidgit/Torch-nn-ext",
	license = "GPL"
}

dependencies = {
	"torch >= 7.0",
	"nn >= 1.0"
}

build = {
	type = "command",
	build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
	]],
	install_command = "cd build && $(MAKE) install"
}
