/*
   FALCON - The Falcon Programming Language.
   FILE: closeddata.h

   Data for closures
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 10 Jan 2013 14:28:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_CLOSEDDATA_H
#define FALCON_CLOSEDDATA_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/variable.h>
#include <falcon/closeddata.h>

namespace Falcon {

class Class;
class ItemArray;
class Symbol;

/** Data for closures. */
class FALCON_DYN_CLASS ClosedData
{
public:
   ClosedData();
   ClosedData( const ClosedData& other );
   ~ClosedData();

   void copy( const ClosedData& other );
   void gcMark( uint32 mark );
   uint32 gcMark() const { return m_mark; }

   void add( const String& name, const Item& value );
   void add( const Symbol* sym, const Item& value );
   Item* get( const String& name ) const;
   Item* get( const Symbol* sym ) const;
   ClosedData* clone() const { return new ClosedData(*this); }
   uint32 size() const;

   void flatten( VMContext* ctx, ItemArray& subItems ) const;
   void unflatten( VMContext* ctx, ItemArray& subItems, uint32 pos = 0 );

   Class* handler() const;

   void defineSymbols( VMContext* ctx );

private:
   class Private;
   ClosedData::Private* _p;

   uint32 m_mark;
};

}

#endif

/* end of closeddata.h */
