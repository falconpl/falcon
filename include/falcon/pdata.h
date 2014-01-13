/*
   FALCON - The Falcon Programming Language.
   FILE: pdata.h

   Engine, VM Process or processor specific persistent data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 13 Jan 2014 16:22:03 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PDATA_H_
#define _FALCON_PDATA_H_

#include <falcon/setup.h>

namespace Falcon  {
class Item;
class String;

/** Engine, VM Process or processor specific persistent data.
 *
 */
class FALCON_DYN_CLASS PData
{
public:
   PData();
   ~PData();

   /** Adds or sets persistent data. */
   bool set( const String& id, const Item& data )const;

   /** Get Engine-level persistent data.
   * \return false if the object is not found
   * \param bPlaceHolder if true, will generate a new entry for the ID, but still return false.
   *
   * If bPlaceHolder is true, and the required item is not present,
   * the first invocation will return false, nevertheless an empty item will be
   * added at given ids; in this way, subsequent calls will return true, atomically.
   */
   bool get( const String& id, Item& data, bool bPlaceHolder=false ) const;

   /* Remove Engine-level persistent data */
   bool remove( const String& id ) const;

private:
   class Private;
   Private *_p;

};

}

#endif

/* end of pdata.h */


