/*
   FALCON - The Falcon Programming Language.
   FILE: pcode.h

   Utilities to manage the Falcon Virtual Machine pseudo code.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 24 Jul 2009 19:42:40 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_PCODE_H_
#define FALCON_PCODE_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/pcodes.h>

namespace Falcon
{

class Module;

class FALCON_DYN_SYM PCODE
{
public:
   /** Rotates endianity in IDs and numbers inside the PCode.
    *
    * \param code the raw pcode sequence
    * \param codeSize the size in bytes of the code sequence.
    * \param into true to actually endianize the code.
    */
   static void deendianize( byte* code, uint32 codeSize, bool into = false );

   /** Rotates endianity in IDs and numbers inside the PCode.
    *
    * \param code the raw pcode sequence
    * \param codeSize the size in bytes of the code sequence.
    */
   static void endianize( byte* code, uint32 codeSize )
   {
       deendianize( code, codeSize, true );
   }

   static void convertEndianity( uint32 paramType, byte* targetArea, bool into=false );
   static uint32 advanceParam( uint32 paramType );
};

}

#endif /* FALCON_PCODE_H_ */

/* end of pcode.h */
