/*
   FALCON - The Falcon Programming Language.
   FILE: bufext_st.h

   Buffering extensions
   Interface extension functions
   -------------------------------------------------------------------
   Author: Maximilian Malek
   Begin: Sun, 20 Jun 2010 18:59:55 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: The above AUTHOR

         Licensed under the Falcon Programming Language License,
      Version 1.1 (the "License"); you may not use this file
      except in compliance with the License. You may obtain
      a copy of the License at

         http://www.falconpl.org/?page_id=license_1_1

      Unless required by applicable law or agreed to in writing,
      software distributed under the License is distributed on
      an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
      KIND, either express or implied. See the License for the
      specific language governing permissions and limitations
      under the License.

*/

/** \file
   String table - export internationalizable stings to both C++
                  and Falcon modules.
*/

//WARNING: the missing of usual #ifndef/#define pair
//         is intentional!

// See the contents of this header file for a deeper overall
// explanation of the MODSTR system.
#include <falcon/message_defs.h>

// The first parameter is an unique identifier in your project that
// will be bound to the correct entry in the module string table.
// Falcon::VMachine::moduleString( bufext_msg_1 ) will
// return the associated string or the internationalized version.
// FAL_STR( bufext_msg_1 ) macro can be used in standard
// functions as a shortcut.

#define FAL_STR_bufext_inv_read "Tried to read beyond valid buffer space"
#define FAL_STR_bufext_inv_write "Tried to write beyond valid buffer space"
#define FAL_STR_bufext_buf_full "Buffer is full; can't write more data"

FAL_MODSTR( bufext_inv_endian,            "Invalid endian ID" );
FAL_MODSTR( bufext_bytebuf_fixed_endian,  "This ByteBuf has a fixed endian, can not be changed" );
FAL_MODSTR( bufext_not_buf,               "Unsupported buffer type or not a buffer" );
FAL_MODSTR( bufext_inv_charsize,          "Invalid char size, must be 1, 2, or 4" );
FAL_MODSTR( bufext_bitbuf_nofloat,        "BitBuf does not support reading/writing floating-point" );
FAL_MODSTR( bufext_inv_read,              FAL_STR_bufext_inv_read);
FAL_MODSTR( bufext_inv_write,             FAL_STR_bufext_inv_write );
FAL_MODSTR( bufext_buf_full,              FAL_STR_bufext_buf_full );

/* end of bufext_st.h */
