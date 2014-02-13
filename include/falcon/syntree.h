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
#include <falcon/treestep.h>

namespace Falcon
{

class Statement;
class SymbolMap;
class Symbol;
class Expression;
class ClassSynTree;

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
class FALCON_DYN_CLASS SynTree: public TreeStep
{

public:
   SynTree( int line = 0, int chr = 0);
   SynTree( const SynTree& other );
   virtual ~SynTree();

   /** Direct interface. 
    Faster than nth()*/
   TreeStep* at( int pos ) const;
   /** Direct interface. 
    Faster than arity()*/
   size_t size() const;
 
   bool empty() const;
   TreeStep* first() const;
   TreeStep* last() const;
   
   /** Returns true if the expression can be found alone in a statement. */
   inline virtual bool isStandAlone() const { return true; }

   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual SynTree* clone() const { return new SynTree(*this); }


   virtual int32 arity() const;   
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   virtual bool insert( int32 pos, TreeStep* element );
   virtual bool remove( int32 pos );
   virtual TreeStep* detach( int32 pos );
   virtual void clear();
   
   /** Returns the selector expression for this block.

    Conditional blocks have selector expression determining whether the 
    block should be entered or not.
    */   
   virtual TreeStep* selector() const { return m_selector; }
   /** Changes the selector for this block
    \param e A new unparented selector expression.

    Conditional blocks have selector expression determining whether the 
    block should be entered or not.
    */  
   virtual bool selector( TreeStep* e );

      
   /** Appends a statement.
    The method will silently fail if the step has already a parent.
    */
   virtual bool append( TreeStep* step );

   static void apply_( const PStep* ps, VMContext* ctx );

   /** Returns the symbol table for this block.
    \param bmake if true, generate a table if not already created.

    This method returns (and eventually creates) a symbol table
    that can be used to store variable names local to this block.
    */
   SymbolMap* locals( bool bmake = true );

   /** Gets the head symbol for this syntree.
    \return A previously set head symbol or 0.
    
    Some syntree have a meaningful "head" symbol that is used in different
    contexts to identify the syntree. For instance, it can be used in the 
    catch clause as the symbol where the incoming raised item should be stored.
    
    This is an extra space in the syntree where this information can be stored.
    */
   const Symbol* target() const { return m_head; }
   
   /** Gets the head symbol for this syntree.
    \param s The symbol to be set.
    
    Some syntree have a meaningful "head" symbol that is used in different
    contexts to identify the syntree. For instance, it can be used in the 
    catch clause as the symbol where the incoming raised item should be stored.
    
    This is an extra space in the syntree where this information can be stored.
    
    \note The ownership of the symbol stays on the caller.
    */
   void target( const Symbol* s );
      

protected:   
   class Private;
   Private* _p;
   
   const Symbol* m_head;
   TreeStep* m_selector;

   friend class ClassSynTree;
};

}

#endif

/* end of syntree.h */

