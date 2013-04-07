/*
   FALCON - The Falcon Programming Language.
   FILE: expristring.h

   Syntactic tree item definitions -- expression elements -- i-string
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Mar 2013 20:23:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRISTRING_H
#define FALCON_EXPRISTRING_H

#include <falcon/setup.h>
#include <falcon/expression.h>
#include <falcon/item.h>
#include <falcon/mt.h>

namespace Falcon {

class Stream;
class VMachine;

/** Expression holding an international string value.
 *
 * The expression automatically tries to resolve the string
 * in the process translation table, and keeps
 * resolved results safe via a garbage token.
 */
class FALCON_DYN_CLASS ExprIString: public Expression
{
public:
   ExprIString( int line = 0, int chr = 0 );
   ExprIString( const String& orig, int line = 0, int chr = 0 );
   ExprIString( const ExprIString& other );

   virtual ~ExprIString();

   virtual bool simplify( Item& result ) const;

   static void apply_( const PStep* s1, VMContext* ctx );

   /** Returns the resolved version of this string.
    *
    * This tries to resolve the string in the process translation table, or
    * returns the already resolved string if possible.
    *
    * After this call, this string might be atomically changed across the engine,
    * but the value of the old resolved string is preserved and left to the GC
    * for cleanup when necessary.
    *
    * \note It's safe to call this method from multiple threads.
    */
   String* resolve( VMContext* ctx ) const;

   /** Changes the stored original string.
    */
   void original( const String& orig );
   const String& original() const;

   virtual ExprIString* clone() const;
   virtual void render( TextWriter* tw, int depth ) const;
   virtual bool isStandAlone() const;

private:
   mutable uint32 m_tlgen;
   String m_original;
   mutable String* m_resolved;
   mutable GCLock* m_lock;
   mutable Mutex m_mtx;
};

}

#endif

/* end of expristring.h */
