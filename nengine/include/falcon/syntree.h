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
class Symbol;

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
   static void apply_empty_( const PStep* ps, VMContext* ctx );

   virtual void describeTo( String& tgt ) const;

   /** Returns the symbol table for this block.
    \param bmake if true, generate a table if not already created.

    This method returns (and eventually creates) a symbol table
    that can be used to store variable names local to this block.
    */
   SymbolTable* locals( bool bmake = true );

   /** Gets the head symbol for this syntree.
    \return A previously set head symbol or 0.
    
    Some syntree have a meaningful "head" symbol that is used in different
    contexts to identify the syntree. For instance, it can be used in the 
    catch clause as the symbol where the incoming raised item should be stored.
    
    This is an extra space in the syntree where this information can be stored.
    */
   Symbol* headSymbol() const { return m_head; }
   
   /** Gets the head symbol for this syntree.
    \param s The symbol to be set.
    
    Some syntree have a meaningful "head" symbol that is used in different
    contexts to identify the syntree. For instance, it can be used in the 
    catch clause as the symbol where the incoming raised item should be stored.
    
    This is an extra space in the syntree where this information can be stored.
    
    \note The ownership of the symbol stays on the caller.
    */
   void headSymbol( Symbol* s ) { m_head = s; }
   
protected:
   class Private;
   Private* _p;
   
   SymbolTable* m_locals;
   Statement* m_single;
   Symbol* m_head;
};

}

#endif

/* end of syntree.h */

