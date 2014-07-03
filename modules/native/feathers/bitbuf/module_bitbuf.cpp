/*
   FALCON - The Falcon Programming Language.
   FILE: module_bitbuf.cpp

   Buffering extensions
   Main module entity
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 08 Jul 2013 13:22:03 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

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

#include <falcon/class.h>
#include <falcon/ov_names.h>
#include "bitbuf_ext.h"
#include "bitbuf_mod.h"
#include "buffererror.h"

#include "module_bitbuf.h"

namespace Falcon {
namespace Feathers {

ModuleBitbuf::ModuleBitbuf():
         Module("bitbuf", true)
{
   this->addConstant( "NATIVE_ENDIAN", (Falcon::int64)Falcon::Ext::BitBuf::e_endian_same );
   this->addConstant( "LITTLE_ENDIAN", (Falcon::int64)Falcon::Ext::BitBuf::e_endian_little );
   this->addConstant( "BIG_ENDIAN",    (Falcon::int64)Falcon::Ext::BitBuf::e_endian_big );
   this->addConstant( "REVERSE_ENDIAN",(Falcon::int64)Falcon::Ext::BitBuf::e_endian_reverse );

   Falcon::Class *bitbuf = Falcon::Ext::init_classbitbuf();

   this->addMantra( bitbuf, true );
   this->addMantra( new Falcon::Ext::ClassBitBufError, true );
}

}
}

