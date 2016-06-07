Intera SDK
==============

The Intera SDK provides a platform for development of custom applications for Intera Robots.

This repository contains metapackages and files for installation/use of the Intera SDK.
Additionally, this repositories contains the Python interface classes and examples for
action servers and control of the Intera Robot from Rethink Robotics.

Installation
------------
| Please follow the Getting Started wiki page for instructions on installation of the Intera SDK:
| http://sdk.rethinkrobotics.com/wiki/Workstation_Setup

Code & Tickets
--------------

+-----------------+----------------------------------------------------------------+
| Documentation   | http://sdk.rethinkrobotics.com/wiki                            |
+-----------------+----------------------------------------------------------------+
| Issues          | https://github.com/RethinkRobotics/intera_sdk/issues           |
+-----------------+----------------------------------------------------------------+
| Contributions   | http://sdk.rethinkrobotics.com/wiki/Contributions              |
+-----------------+----------------------------------------------------------------+

Intera Repository Overview
--------------------------

::

     .
     |
     +-- intera_sdk/          intera_sdk metapackage containing all intera sdk packages
     |
     +-- intera_interface     Python API for communicating with Intera-enabled robots
     |   +-- cfg/
     |   +-- scripts/ 
     |   +-- src/
     |
     +-- intera_example       examples using the Python API for Intera-enable robots
     |   +-- cfg/
     |   +-- scripts/ 
     |   +-- src/
     |
     +-- intera.sh            convenient environment initialization script


Other Intera Repositories
-------------------------
+------------------+-----------------------------------------------------+
| intera_common    | https://github.com/RethinkRobotics/intera_common    |
+------------------+-----------------------------------------------------+

Latest Release Information
--------------------------

http://sdk.rethinkrobotics.com/wiki/Release-Changes