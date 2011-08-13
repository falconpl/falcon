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
#include <falcon/pstep.h>

namespace Falcon
{

class Statement;
class SymbolTable;

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
class FALCON_DYN_CLASS SynTree: public PStep
{

public:
   SynTree();
   virtual ~SynTree();

   int size() const;
   bool empty() const;

   Statement* first() const;
   Statement* last() const;
   Statement* at( int pos ) const;
   void set( int pos, Statement* p );

   void insert( int pos, Statement* step );
   void remove( int pos );
   SynTree& append( Statement* step );

   static void apply_( const PStep* ps, VMContext* ctx );
   static void apply_single_( const PStep* ps, VMContext* ctx );

   virtual void describeTo( String& tgt ) const;

   /** Returns the symbol table for this block.
    \param bmake if true, generate a table if not already created.

    This method returns (and eventually creates) a symbol table
    that can be used to store variable names local to this block.
    */
   SymbolTable* locals( bool bmake = true );

protected:
   class Private;
   Private* _p;
   
   SymbolTable* m_locals;
   Statement* m_single;
};

}

#endif

/* end of syntree.h */

