BUILD_DIR="build"

git submodule init

first_build=false

for OPTION in "$@"
do
    case "${OPTION}"
    in
        --setup) ;&
        -s) first_build=true;
        rm -rf "$BUILD_DIR";;
    esac
done

if [ ! -d "$BUILD_DIR" ]; then
	echo "Creating build directory!";
	mkdir "$BUILD_DIR";
	first_build=true
fi

cd "$BUILD_DIR"

if [ "$first_build" = true ] ; then
    cmake ..
fi
cmake --build .
