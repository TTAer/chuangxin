SparkFun RFID USB-C Reader
========================================

[![SparkFun RFID USB-C Reader](https://cdn.sparkfun.com/assets/parts/2/9/5/9/5/28020-RFID-USB-C-Reader-feature.jpg)](https://www.sparkfun.com/sparkfun-rfid-usb-c-reader.html)
[*SparkFun RFID USB-C Reader (SEN-28020)*](https://www.sparkfun.com/sparkfun-rfid-usb-c-reader.html)


The SparkFun RFID USB-C Reader is a simple to use, USB-to-serial breakout board for the [ID-Innovations Tiny, non-write, RFID modules](https://www.id-innovations.com/Modules(non%20write).htm) *(excluding the Mifare readers, marked with an **MF**)*. This board features a USB-C connector with a CH340C USB-to-serial chip; and comes with an LED and buzzer to indicate card reads, which can be disabled if necessary.

With the exclusion of the ID-3XX RFID modules, which requires an external antenna, users only need to attach an RFID reader module onto the board's 2mm headers and connect a USB-C cable to their computer to get started. To read compatible RFID tags, users will need to open a terminal emulation program with the serial port configured to **9600bps** 8-N-1; then, scan the tag and its unique ID will be displayed.


> [!WARNING]
> This product does not come with an RFID module. We offer these compatible modules in our catalog:
>
> - [ID-3LA](https://www.sparkfun.com/rfid-reader-id-3la-125-khz.html)
> - [ID-12LA](https://www.sparkfun.com/rfid-reader-id-12la-125-khz.html)
> - [ID-20LA](https://www.sparkfun.com/rfid-reader-id-20la-125-khz.html)


> [!IMPORTANT]
> While most of the [ID-innovations read-only Tiny modules](https://www.id-innovations.com/Modules(non%20write).htm), will have an internal antenna; any of the ID-3XX modules will likely require an external antenna to function. Users should consult with the module's datasheet, when purchasing.


Documentation
--------------
- **[Hookup Guide (mkdocs)](http://docs.sparkfun.com/SparkFun_RFID_USB-C_Reader/)** - The hookup guide for the SparkFun RFID USB-C Reader hosted by GitHub pages.<br>
  [![Built with Material for MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/) [![GitHub Pages Deploy](https://github.com/sparkfun/SparkFun_RFID_USB-C_Reader/actions/workflows/build_documentation.yml/badge.svg)](https://github.com/sparkfun/SparkFun_RFID_USB-C_Reader/actions/workflows/build_documentation.yml)

  *Need to download or print our hookup guide?*

  - [Print *(Print to PDF)* from Single-Page View](http://docs.sparkfun.com/SparkFun_RFID_USB-C_Reader/print_view)


Repository Contents
-------------------
- **[/docs](/docs/)** - Online documentation files
    - [assets](/docs/assets/) - Assets files
        - [3d_model](/docs/assets/3d_model/) - Files for the 3D models
            - [3D CAD Model](/docs/assets/3d_model/cad_model.step) (.step)
        - [board_files](/docs/assets/board_files/) - Files for the product design
            - [KiCad Design Files](/docs/assets/board_files/kicad_files.zip) (.zip)
            - [Schematic](/docs/assets/board_files/schematic.pdf) (.pdf)
            - [Dimensions](/docs/assets/board_files/dimensions.pdf) (.pdf)
        - [component_documentation](/docs/assets/component_documentation/) - Datasheets for hardware components
        - [img/hookup_guide/](/docs/assets/img/hookup_guide/) - Images for hookup guide documentation
- **[/Hardware](/Hardware/)** - Hardware design files (.brd, .sch)
  - **[/Production](/Production/)** - Production files


Product Variants
----------------
- [SEN-28020](https://www.sparkfun.com/sparkfun-rfid-usb-c-reader.html) - USB-C connector and CH340C converter
- [SEN-09963](https://www.sparkfun.com/sparkfun-rfid-usb-reader.html) - Original product *(w/ USB mini-B connector and FTDI converter)*


Version History
---------------
- [v10](https://github.com/sparkfun/SparkFun_RFID_USB-C_Reader/releases/tag/v10) - Initial Release


License Information
-------------------

This product is ***open source***!

Please review the [`LICENSE.md`](./LICENSE.md) file for license information.

If you have any questions or concerns about licensing, please contact technical support on our [SparkFun forums](https://forum.sparkfun.com/viewforum.php?f=152).

Distributed as-is; no warranty is given.

- Your friends at SparkFun.
