/*
   FALCON - The Falcon Programming Language.
   FILE: syntree.h

   Syntactic tree item definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_SYNTREE_H
#define FALCON_SYNTREE_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>
#include <falcon/pstep.h>

#include <vector>

namespace Falcon
{

/** Syntactic tree.
 *
 * The syntactic tree is actually a list of PStep, that may be either
 * (standalone) expressions or statements, which may hold other syntactic
 * sub-trees.
 *
 * The syntactic tree is a PStep on itself. This means that can be seen
 * as a PCode that is directly executed by the virtual machine.
 *
 * The effect of executing a Syntree (that is, calling its apply() methid)
 * is that of calling the perform() method on all the PStep it holds, in sequence.
 *
 * \note None of the methods in this class is guarded. Accessing any invalid
 * item outside 0..size() will cause crash.
 */
class SynTree: public PStep
{

public:
   SynTree();
   virtual ~SynTree();

   int size() const { return m_steps.size(); }

   PStep* first() { return m_steps.first(); }
   PStep* last()  { return m_steps.last(); }
   PStep* at( int pos ) const { return m_steps[pos]; }
   void set( int pos, PStep* p )  { delete m_steps[pos]; m_steps[pos] = p; }

   void insert( int pos, const PStep* step ) { m_steps.insert( m_steps.begin()+pos, step ); }
   void remove( int pos ) {
      PStep* p = m_steps[ m_steps.begin()+pos ];
      m_steps.remove( m_steps.begin()+pos );
      delete p;
   }

   void append( const PStep* step ) { m_steps.append( step ); }

   virtual void perform( VMachine* vm ) const;
   virtual void apply( VMachine* vm ) const;
   void toString( String& tgt ) const;

private:

   std::vector<PStep*> m_steps;
};

}

#endif

/* end of syntree.h */

