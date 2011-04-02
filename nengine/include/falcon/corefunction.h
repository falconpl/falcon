/*
   FALCON - The Falcon Programming Language.
   FILE: corefunction.h

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_COREFUNCTION_H_
#define FALCON_COREFUNCTION_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

class Function;
class Module;

/**
 Class handling a function as an item in a falcon script.
 */

class FALCON_DYN_CLASS CoreFunction: public Class
{
public:

   class cpars {
   public:
      cpars( const String& name, Module* mod ):
         m_name( name ),
         m_module( mod )
         {}

      const String& m_name;
      Module* m_module;
   };

   CoreFunction();
   virtual ~CoreFunction();

   virtual void* create(void* creationParams ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   virtual void describe( void* instance, String& target ) const;
};

}

#endif /* FUNCTION_H_ */

/* end of corefunction.h */
