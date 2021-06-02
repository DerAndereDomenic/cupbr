# CUPBR
<img title="Sample render" alt="Alt text" src="sample.bmp" width="60%">

# Building
This project is maintained and tested using Arch Linux. In principle this also runs under windows but is tested less frequently. We use git submodules to manage dependencies like:
<ul>
    <li> <a href="https://www.glfw.org">glfw</a>
    <li> <a href="https://glad.dav1d.de">glad</a>
    <li> <a href="https://github.com/leethomason/tinyxml2">tinyxml2</a>
    <li> <a href="https://github.com/nothings/stb">stb</a>
</ul>
so there is no need to install them. You can either clone the repository recursively using

```
git clone --recurse-submodules https://github.com/DerAndereDomenic/cupbr.git
```

or clone it as usual. The attached build script will update the submodules in this case.<br>
To build the project simply run

```
sh build.sh
```

This will create the CUPBR executable inside the **/bin** folder. After that, running the script will act like **cmake --build build**. To do redo a complete setup, run

```
sh build.sh -s
```

# Usage
After building the project the CUPBR executable is located in the **/bin** folder. When executing the file, a sample scene is rendered using path tracing. To render another scene you can pass the path to the scene file as an argument

```
./bin/CUPBR res/Scene/<Scene-File>
```

**Controls**<br>
<ul>
    <li> Escape: Close the application.
    <li> Alt: Lock / Unlock the camera.
    <li> If the camera is unlocked WASD and mouse movement can be used to navigate the scene.
    <li> M: Open / Close Rendering Settings.
</ul>