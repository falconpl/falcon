/*
   FALCON - The Falcon Programming Language.
   FILE: falconinstance_private.h

   Instance of classes declared in falcon scripts -- private part
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 24 Sep 2013 16:06:37 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_INSTANCE_PRIVATE_H_
#define _FALCON_INSTANCE_PRIVATE_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/delegatemap.h>
#include <map>

namespace Falcon
{

/** Class common to falcon instance and its friend FalconClass */
class FalconInstance::Private
{
public:
   typedef std::map< const String*, Item, StringPtrCmp > Data;

   Data m_data;
   const FalconClass* m_origin;
   uint32 m_mark;
   DelegateMap m_delegates;

   Private():
      m_origin(0),
      m_mark(0)
   {}

   Private( const Private& p ):
      m_data(p.m_data),
      m_origin(p.m_origin),
      m_mark(p.m_mark)
   {}


   inline Item* getProperty( const String* b )
   {
      Private::Data::iterator prop = m_data.find(b);
      if( prop != m_data.end() )
      {
         return &prop->second;
      }

      return 0;
   }
};

}

#endif

/* end of falconinstance_private.h */
