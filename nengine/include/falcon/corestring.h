/*
   FALCON - The Falcon Programming Language.
   FILE: corestring.h

   String object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 15:11:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CORESTRING_H_
#define _FALCON_CORESTRING_H_

#include <falcon/setup.h>
#include <falcon/class.h>
#include <falcon/string.h>

namespace Falcon
{

/**
 Class handling a string as an item in a falcon script.
 */

class FALCON_DYN_CLASS CoreString: public Class
{
public:

   class cpars {
   public:
      cpars( const String& other, bool bufferize = false ):
         m_other( other ),
         m_bufferize(bufferize)
      {}

      const String& m_other;
      bool m_bufferize;
   };

   CoreString();
   virtual ~CoreString();

   virtual void* create( void* creationParams ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( Stream* stream, void* self ) const;
   virtual void* deserialize( Stream* stream ) const;

};

}

#endif /* _FALCON_CORESTRING_H_ */

/* end of corestring.h */
