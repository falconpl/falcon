/*
   FALCON - The Falcon Programming Language.
   FILE: zlib_st.h

   ZLib module - String table.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 May 2008 22:56:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   ZLib module - String table.
*/

#ifndef FLC_ZLIB_ST_H
#define FLC_ZLIB_ST_H

#include <falcon/module.h>

FAL_MODSTR( zl_msg_nomem, "Not enough memory" );
FAL_MODSTR( zl_msg_noroom, "Not enough room in output buffer to decompress");
FAL_MODSTR( zl_msg_invformat, "Data supplied is not in compressed format");
FAL_MODSTR( zl_msg_generic, "An unknown uncompress error has occurred");
FAL_MODSTR( zl_msg_vererr, "Data compressed using incompatible ZLib version");
FAL_MODSTR( zl_msg_notct, "Data was not compressed with ZLib.compressText" );

#endif

/* zlib_st.h */
