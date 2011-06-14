/*
   FALCON - The Falcon Programming Language.
   FILE: endianity.h

   Support for endian-transformation of basic data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 11:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_ENDIANITY_H_
#define _FALCON_ENDIANITY_H_

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

/** Converts an unaligned buffer in an unsigned int, 16 bit */
inline uint16 getUInt16( byte* buf )
{
   union t_data {
      uint16 ui16;
      byte b[2];
   } data;

   data.b[0] = buf[0];
   data.b[1] = buf[1];
   return data.ui16;
}

/** Converts an unaligned buffer in an unsigned int, 16 bit, reversing its endianity */
inline uint16 getUInt16Reverse( byte* buf )
{
   union t_data {
      uint16 ui16;
      byte b[2];
   } data;

   data.b[1] = buf[0];
   data.b[0] = buf[1];
   return data.ui16;
}


/** Converts an unaligned buffer in an unsigned int, 32 bit */
inline uint32 getUInt32( register byte* buf )
{
   union t_data {
      uint32 ui32;
      byte b[4];
   } data;

   register byte* dtb = data.b;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf;
   
   return data.ui32;
}

/** Converts an unaligned buffer in an unsigned int, 32 bit, reversing its endianity */
inline uint32 getUInt32Reverse( register byte* buf )
{
   union t_data {
      uint32 ui32;
      byte b[4];
   } data;

   register byte* dtb = data.b+3;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf;

   return data.ui32;
}

/** Converts an unaligned buffer in an unsigned int, 64 bit */
inline uint64 getUInt64( register byte* buf )
{
   union t_data {
      uint64 ui64;
      byte b[8];
   } data;

   register byte* dtb = data.b;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;

   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf;

   return data.ui64;
}

/** Converts an unaligned buffer in an unsigned int, 64 bit, reversing its endianity */
inline uint64 getUInt64Reverse( register byte* buf )
{
   union t_data {
      uint64 ui64;
      byte b[8];
   } data;

   register byte* dtb = data.b+7;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;

   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf;

   return data.ui64;
}


/** Converts an unaligned buffer in a 32 bit float */
inline float getFloat32( register byte* buf )
{
   union t_data {
      float f32;
      byte b[4];
   } data;

   register byte* dtb = data.b;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf;

   return data.f32;
}


/** Converts an unaligned buffer in a 32 bit float, reversing its endianity */
inline float getFloat32Reverse( register byte* buf )
{
   union t_data {
      float f32;
      byte b[4];
   } data;

   register byte* dtb = data.b+3;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf;

   return data.f32;
}

/** Converts an unaligned buffer in a 64 bit float */
inline double getFloat64( register byte* buf )
{
   union t_data {
      double f64;
      byte b[8];
   } data;

   register byte* dtb = data.b;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;

   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf; ++dtb; ++buf;
   *dtb = *buf;

   return data.f64;
}


/** Converts an unaligned buffer in a 64 bit float, reversing its endianity */
inline double getFloat64Reverse( register byte* buf )
{
   union t_data {
      double f64;
      byte b[8];
   } data;

   register byte* dtb = data.b+7;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;

   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf; --dtb; ++buf;
   *dtb = *buf;

   return data.f64;
}

/** reverse the endianness of arrays containing 16 bit words */
inline void REArray_16Bit( register byte* data, register length_t count )
{
   while (count > 0)
   {
      register byte b = *data;
      data[1] = *data;
      ++data;
      *data = b;
      ++data;
      
      --count;
   }
}

/** reverse the endianness of arrays containing 32 bit words */
inline void REArray_32Bit( register byte* data, register length_t count )
{
   while (count > 0)
   {
      register byte b0 = *data;
      register byte b1 = data[1];
      register byte b2 = data[2];
      *data = data[3];
      ++data;
      *data = b2;
      ++data;
      *data = b1;
      ++data;
      *data = b0;
      
      --count;
   }
}

}

#endif

/* end of endianity.h */
