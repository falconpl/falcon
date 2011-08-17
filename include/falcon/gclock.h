/*
   FALCON - The Falcon Programming Language.
   FILE: gclock.h

   Lock preventing an item to be garbage collected.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 Jan 2011 11:55:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef FALCON_GLOCK_H
#define	FALCON_GLOCK_H

#include <falcon/item.h>

namespace Falcon
{

class Collector;

/** Object representing a garbage lock.
 *
 * This object indicates that an item is assigned to the garbabge collector,
 * but it is temporarily unavailable for collection.
 *
 * The lock can be given to different holders and the disposed once the holders
 * are gone.
 *
 * While a GCLock for an item exists, the item gets marked at each loop (and thus,
 * deep items have their components marked as well).
 *
 * The lock can only be created via Collector::lock(). Constructors and destructors
 * are reserved to the Collector class.
 */

class GCLock
{
public:

    /** Gets the item locked by this lock. 
     * @note This method should be accessed only as const,
     * as the GC may be scanning right now.
     */
    inline const Item& item() const { return m_item; }

    /** Marks this garbage lock as disposeable.
     *
     * After calling this method, the pointer to this object must be
     * considered invalid.
     */
    void dispose() { m_bDisposed = true; }

protected:
    GCLock( const Item& orig ):
        m_item( orig ),
        m_bDisposed( false )
    {}

    GCLock():
        m_bDisposed( false )
    {}

    ~GCLock() {}
    
private:
    Item m_item;
    GCLock* m_next;
    GCLock* m_prev;
    volatile bool m_bDisposed;

    friend class Collector;
};

}

#endif	/* FALCON_GCTOKEN_H */

/* end of gctoken.h */
