/*
   FALCON - The Falcon Programming Language.
   FILE: falconinstance.cpp

   Instance of classes declared in falcon scripts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 14:35:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/falconinstance.h>
#include <falcon/falconclass.h>
#include <falcon/psteps/exprinherit.h>
#include <falcon/stderrors.h>
#include <falcon/string.h>

#include "falconinstance_private.h"

namespace Falcon
{



FalconInstance::FalconInstance()
{
   _p = new Private;
}

FalconInstance::FalconInstance( const FalconClass* origin )
{
   _p = new Private();
   _p->m_origin = origin;
}


FalconInstance::FalconInstance( const FalconInstance& other )
{
   _p = new Private(*other._p);
}

FalconInstance::~FalconInstance()
{
   delete _p;
}


const FalconClass* FalconInstance::origin() const
{
   return _p->m_origin;
}


Item* FalconInstance::getProperty_internal( const String* name ) const
{
   return _p->getProperty( name );
}


bool FalconInstance::getProperty( const String& name, Item& target ) const
{
   // fast path: return already cached property
   Item* itm = _p->getProperty( &name );
   if( itm != 0 )
   {
      target.copyFromRemote(*itm);
      return true;
   }
   return false;
}


bool FalconInstance::setProperty( const String& name, const Item& value )
{
   Item* itm = _p->getProperty( &name );
   if( itm != 0 )
   {
      itm->copyFromLocal(value);
      return true;
   }
   return false;
}


void FalconInstance::gcMark( uint32 mark )
{
   if( mark != _p->m_mark )
   {
      _p->m_mark = mark;
      Private::Data::iterator iter = _p->m_data.begin();
      Private::Data::iterator end = _p->m_data.end();
      while( iter != end )
      {
         Item& item = iter->second;
         item.gcMark( mark );
         ++iter;
      }
   }
}


uint32 FalconInstance::currentMark() const
{
   return _p->m_mark;
}

void FalconInstance::makeStorageData( ItemArray& array ) const
{
   // trust the strict order granted by std::map.
   // we should re-oprdinate properties in classes in case we change structure to an unordered one.

   Private::Data::const_iterator iter = _p->m_data.begin();
   Private::Data::const_iterator end = _p->m_data.end();
   while( iter != end )
   {
      const Item& item = iter->second;
      item.lock();
      array.append(item);
      item.unlock();
      ++iter;
   }
}


void FalconInstance::restoreFromStorageData( ItemArray& array )
{
   Private::Data::iterator iter = _p->m_data.begin();
   Private::Data::iterator end = _p->m_data.end();
   uint32 pos = 0;
   while( iter != end && pos < array.length() )
   {
      Item& src = array[pos++];
      Item& item = iter->second;
      item.lock();
      item = src;
      item.unlock();
      ++iter;
   }
}


}

/* end of falconinstance.cpp */
