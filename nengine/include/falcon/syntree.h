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

class Statement;

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

   Statement* first() { return m_steps.front(); }
   Statement* last()  { return m_steps.back(); }
   Statement* at( int pos ) const { return m_steps[pos]; }
   void set( int pos, Statement* p );

   void insert( int pos, Statement* step ) {
      m_steps.insert( m_steps.begin()+pos, step );
   }

   void remove( int pos );

   SynTree& append( Statement* step ) {
      m_steps.push_back( step );
      return *this;
   }

   virtual void apply( VMachine* vm ) const;
   virtual void toString( String& tgt ) const;

   inline const String toString() const
   {
      String temp;
      toString( temp );
      return temp;
   }

private:

   std::vector<Statement*> m_steps;
};

}

#endif

/* end of syntree.h */

