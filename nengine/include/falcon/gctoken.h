/*
   FALCON - The Falcon Programming Language.
   FILE: gctoken.h

   This is the representation of an item in the garbage collector.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 Jan 2011 11:55:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef FALCON_GCTOKEN_H
#define	FALCON_GCTOKEN_H

#include <falcon/class.h>

namespace Falcon
{

class Collector;

/** Data carrier for the garbage collector.
 *
 * This class is used internally by the engine to deliver an item to the garbatge
 * collector.
 *
 * It requires a class which knows how to handle the garbage value, and the
 * value itself as a void*.
 *
 */
class GCToken
{
public:
    GCToken( CoreClass* cls, void* data ):
            m_cls( cls ),
            m_data( data ),
            m_mark(0)
    {}

    ~GCToken() {}

    void mark(uint32 n) { m_mark = n; m_cls->gcMark( m_data, n ); }
    void dispose() { m_cls->dispose( m_data ); }

private:
    CoreClass* m_cls;
    void* m_data;
    uint32 m_mark;

    GCToken* m_next;
    GCToken* m_prev;

    friend class Collector;
};

}

#endif	/* GCTOKEN_H */
