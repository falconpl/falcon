/*
   FALCON - The Falcon Programming Language.
   FILE: usercarrier.h

   Base class for ClassUser based instance reflection.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 07 Aug 2011 12:11:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_USERCARRIER_H_
#define _FALCON_USERCARRIER_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/item.h>

namespace Falcon {

class ClassUser;
class Item;

class UserCarrier
{
public:
   UserCarrier( uint32  itemcount );   
   UserCarrier( const UserCarrier& other );
   
   virtual ~UserCarrier();
   
   virtual void gcMark( uint32 mark );   
   uint32 gcMark() const { return m_gcMark; }

   virtual UserCarrier* clone();
   
   Item* dataAt( uint32 pos ) const { return m_data + pos; }
   uint32 itemCount() const { return m_itemCount; }
   
private:
   
   Item* m_data;
   uint32 m_itemCount;
   uint32 m_gcMark;
};

}

#endif	/* _FALCON_USERCARRIER_H_ */

/* end of usercarrier.h */
