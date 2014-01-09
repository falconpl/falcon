/*
   FALCON - The Falcon Programming Language.
   FILE: service.h

   Base class to expose functionalities through DLL interface.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 15 Nov 2013 15:31:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SERVICE_H_
#define _FALCON_SERVICE_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon
{

/** Base class to expose functionalities through DLL interface.
 *
 */
class FALCON_DYN_CLASS Service
{
public:

   Service( const String& name, Module* master ):
      m_name(name),
      m_module( master )
   {}

   virtual ~Service() {}

   const String& name() const { return m_name; }
   Module* module() const { return m_module; }

   /** Place the underlying data for this service in a item.
    * \param target The item where to place the instance of the underlying data.
    */
   virtual void itemize( Item& target ) const = 0;

private:
   String m_name;
   Module* m_module;
};

}

#endif

/* end of service.h */
