#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: /cmake/install-gnudocs.cmake
#
#   Docs to be installed as part of the GNU-compliancy.
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

set( doc_files
      AUTHORS
      BUILDING
      ChangeLog
      copyright
      LICENSE
      README
      README.editline
      README.mersenne
      TODO
      LICENSE_GPLv2
      RELNOTES
   )

install(
   FILES ${doc_files}
   DESTINATION ${FALCON_SHARE_DIR} )
