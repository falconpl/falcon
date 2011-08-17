/*
   FALCON - The Falcon Programming Language
   FILE: base64.h

   Base64 encoding as per rfc3548
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Jun 2010 15:47:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_BASE64_H_
#define _FALCON_BASE64_H_


#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>

namespace Falcon {

class FALCON_DYN_CLASS Base64
{
public:
   /** Basic implementation */
   static bool encode( byte* data, uint32 dsize, byte* target, uint32& tgsize );

   /** Basic implementation

      \return True if the data could be properly parsed (is a valid Base64 string).
   */
   static bool decode( const String& data, byte* target, uint32& tgsize );

   static void encode( byte* data, uint32 dsize, String& target );
   static void encode( const String& data, String& target );
   static bool decode( const String& data, String& target );

private:
   static byte getBits( uint32 bits );

};

}

#endif /* BASE64_H_ */

/* end of base64 */
