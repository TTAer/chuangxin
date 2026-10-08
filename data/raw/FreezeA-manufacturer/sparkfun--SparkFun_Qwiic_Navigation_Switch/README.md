SparkFun Qwiic Navigational Switch
========================================

[![SparkFun Qwiic Navigational Switch](https://cdn.sparkfun.com/r/600-600/assets/parts/2/8/8/0/0/PRT-27576-Qwiic-Navigation-Switch-Feature.jpg)](https://www.sparkfun.com/sparkfun-qwiic-navigation-switch.html)

[*SparkFun Qwiic Navigational Switch (PRT-27576)*](https://www.sparkfun.com/sparkfun-qwiic-navigation-switch.html)

<!-- INTRODUCTION -->
The SparkFun Navigation Switch adds a small, joystick-like 5-way tactile switch to your I<sup>2</sup>C bus! When connected to a microcontroller, push the 5-way tactile switch toward a direction (up, down, left, right, or center) to provide additional input for your application. The built-in RGB status LED provides visual feedback indicating when a switch is pushed.

<!-- HARDWARE  -->
Populated on the board is a [5-way tactile switch](https://www.sparkfun.com/products/10063). Each direction includes a momentary switch that is read by the PCA9554 8-bit I<sup>2</sup>C I/O expander and notifies the microcontroller when a switch is pushed. The microcontroller will then provide visual feedback by writing to the non-addressable RGB LED. Each channel of the RGB LED can be disconnected from the PCA9554's GPIO and connected to a different input. An interrupt pin is also broken out to notify your microcontroller when a state has changed quickly. We've also included an LED for power. Utilizing our handy Qwiic system, no soldering is required to connect it to the rest of your system. However, we still have broken out 0.1"-spaced pins if you prefer a breadboard.

If you need multiple Qwiic Navigation Switches for your project, fear not! Each board has a configurable I<sup>2</sup>C address, so you can daisy-chain multiple Navigation Switches over Qwiic and address each individually.

<!-- EXAMPLE CODE AND I2C EXPANDER ARDUINO LIBRARY -->
We've written a basic example for the Qwiic Navigation Switch in our GitHub Hardware repository. You will need the [SparkFun I2C Expander Arduino Library](https://github.com/sparkfun/SparkFun_I2C_Expander_Arduino_Library) to read or write to the PCA9554's eight separate I/O pins. You can download the library through the Arduino library manager by searching 'SparkFun I<sup>2</sup>C Expander Arduino Library', or you can get the GitHub repo as a .zip file and install the library from there.

<!-- APPLICATIONS -->
This board is great for projects requiring additional input switches. Add a Qwiic Navigation Switch to your project when navigating a menu, or even add it to a small remote to control a robot!



Repository Contents
-------------------

* **.github/workflows** - YAML files used for GitHub Actions and GitHub Pages/mkdocs
* **/Firmware** - Example code
* **/Hardware** - KiCad design files (.kicad_pcb, .kicad_sch, .kicad_pro)
* **/docs** - Online documentation files
  * **/assets** - Folder containing all the file assets used for product documentation
    * **/3d_model** - Exported 3D models from KiCad
    * **/board_files** - Copy of design files used for product documentation
    * **/component_documentation** - Datasheets and manuals for hardware components
    * **/img** - Images for product documentation
  * **/github** - Files stating how to contribute and filing issues used for product documentation
  * **/javascript** - Folder containing custom javascript used for product documentation
  * **/stylesheet** - Folder containing CSS files used for product documentation
* **/overrides** - Customization files used for product documentation
  * **/.icons** - Icons used for GitHub used for product documentation
  * **./partials** - Used for product documentation



Documentation
--------------
* **[Library](https://github.com/sparkfun/SparkFun_I2C_Expander_Arduino_Library)** - Arduino library for the Qwiic Navigational Switch.
* **[Hookup Guide](https://docs.sparkfun.com/SparkFun_Qwiic_Navigation_Switch/)** - Basic hookup guide for the Qwiic Navigational Switch.



Product Versions
----------------
* [PRT-27576](https://www.sparkfun.com/products/27576) - v1.1



Version History
---------------
* v1.1 - Initial Release


License Information
-------------------

This product is _**open source**_!

Please review the LICENSE.md file for license information.

If you have any questions or concerns on licensing, please contact technical support on our [SparkFun forums](https://forum.sparkfun.com/viewforum.php?f=152).

Distributed as-is; no warranty is given.

- Your friends at SparkFun.

_<COLLABORATION CREDIT>_
