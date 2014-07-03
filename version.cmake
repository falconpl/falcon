#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: version.cmake
#
#   Minimal variables used in the system to identify this version
#
#   This file will be manually edited to mark relase milestones
#
#   -------------------------------------------------------------------
#   Author: Giancarlo Niccolai
#   Begin: Wed, 02 Jul 2014 23:02:17 +0200
#
#   -------------------------------------------------------------------
#   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
#
#   See LICENSE file for licensing details.
#   
#################################################################

# Major.Minor.Revision.Patch
set(FALCON_VERSION_MAJOR 1)
set(FALCON_VERSION_MINOR 0)
set(FALCON_VERSION_REVISION 0)
set(FALCON_VERSION_PATCH 0)

# A symbolic name given to milestone releases
set(FALCON_VERSION_NAME "Peregrine")

# Extra specification given to milestone release (alpha/beta/candidate etc.)
set(FALCON_VERSION_SPEC "alpha")

# Successive build ID after a milestone (number of public + internal releases since major change)
set(FALCON_VERSION_BUILD_ID 0)

# So-name versioning (used to determine compatibility of users with the engine SO).
set(FALCON_SONAME_VERSION 2)
set(FALCON_SONAME_REVISION 0)
set(FALCON_SONAME_AGE 0)


## SONAME and soversion (unix so library informations for engine)
# Remember that SONAME never follows project versioning, but
# uses a VERSION, REVISION, AGE format, where
# VERSION: generational version of the project
# REVISION: times this version has been touched
# AGE: Number of version for which binary compatibility is granted
# In eample, 1.12.5 means that this lib may be dynlinked against
# every program using this lib versioned from 1.8 to 1.12.

