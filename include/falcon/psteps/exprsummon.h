/*
   FALCON - The Falcon Programming Language.
   FILE: exprsummon.h

   Syntactic tree item definitions -- expression elements -- summon
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 18 Apr 2013 15:38:58 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRSUMMON_H
#define FALCON_EXPRSUMMON_H

#include <falcon/setup.h>
#include <falcon/psteps/exprvector.h>
#include <falcon/gclock.h>

namespace Falcon {


/** Base summon expression
 */
class FALCON_DYN_CLASS ExprSummonBase: public ExprVector
{
public:
   ExprSummonBase( int line, int chr, bool isOptional );
   ExprSummonBase( const String& name,  int line, int chr, bool isOptional);
   ExprSummonBase( const ExprSummonBase& other );
   virtual ~ExprSummonBase();

   virtual void render( TextWriter* tw, int32 depth ) const;

   /** Symbols cannot be simplified. */
   inline virtual bool simplify( Item& ) const { return false; }

   const String& message() const { return m_message; }
   void message( const String& msg ) { m_message = msg; }

   virtual TreeStep* selector() const;
   virtual bool selector( TreeStep* ts );

protected:
   TreeStep* m_selector;
   String m_message;
   bool m_bIsOptional;

   static void apply_( const PStep* ps, VMContext* ctx );

   FALCON_DECLARE_INTERNAL_PSTEP(Responded);
   FALCON_DECLARE_INTERNAL_PSTEP_OWNED(Summoned, ExprSummonBase);
};

/** Summon expression.

    a::b[c d e]  => a.b(c d e)
 */
class FALCON_DYN_CLASS ExprSummon: public ExprSummonBase
{
public:
   ExprSummon( int line = 0, int chr = 0 );
   ExprSummon( const String& name,  int line = 0, int chr = 0 );
   ExprSummon( const ExprSummon& other );
   virtual ~ExprSummon();

   inline virtual ExprSummon* clone() const { return new ExprSummon(*this); }
};

/** Optional summon expression.

    It's distinct from the basic summon expression as it's easy to
    represent them in full homoiconic model this way.
 */

class FALCON_DYN_CLASS ExprOptSummon: public ExprSummonBase
{
public:
   ExprOptSummon( int line = 0, int chr = 0 );
   ExprOptSummon( const String& name,  int line = 0, int chr = 0 );
   ExprOptSummon( const ExprSummon& other );
   virtual ~ExprOptSummon();

   inline virtual ExprOptSummon* clone() const { return new ExprOptSummon(*this); }
};

}

#endif

/* end of exprsummon.h */
