/*
   FALCON - The Falcon Programming Language.
   FILE: select.h

   Syntactic tree item definitions -- select statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 13:58:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SELECT_H_
#define _FALCON_SELECT_H_

#include <falcon/statement.h>
#include <falcon/requirement.h>

namespace Falcon {

class Symbol;
class Expression;

/** Handler for the select statement.
 
 The select statements selects a branch depending on the type ID or on the
 class of the result of an expression.
 
 The try statement uses an expressionless select to account for catch clauses.
 For the same reason, branch declarator have a "target symbol" that is not used
 in the "select" statement, but may be used in catch clauses to store the
 incoming caught value.
 */
class FALCON_DYN_CLASS StmtSelect: public Statement
{
public:   
   /** Create the select statement.
    \param expr The expression that will generate the item to be selected, or 0.
    \param line The line where this statement is declared in the source.
    \param chr The character at which this statement is declared in the source.
    
    The expression may be left to 0 if this instance is not meant to be added
    to a syntactic tree directly, but it's used just as a dictionary of
    type selectors.
    
    This is the case of try/catch, but it might be used by third party code
    for similar reasons.
    */
   StmtSelect( Expression* expr = 0, int32 line = 0, int32 chr = 0 );
   virtual ~StmtSelect();

   virtual void describeTo( String& tgt ) const;
   void oneLinerTo( String& tgt ) const;
   
   /** Adds a branch for an integer type ID. 
    \param typeId the type ID that will activate this branch.
    \param block The block that should be activated by that type ID.
    \return True if the entity can be added, false if it was already added.
    
    \note If the \b block is the same as the last one just added, this is
    considered an alternate selector value for the same block. However, the
    search for previously defined blocks is not extended past the last inserted
    block. This means that alternate selectors for the same block must all be
    added one after another.
    
    \note The try/catch statement will use the head symbol in the \b block SynTree
    to store the "catch/in" clause symbol.
    */
   bool addSelectType( int64 typeId, SynTree* block );
   
   /** Adds a branch for a class pointer. 
    \param cls A class.
    \param block The block that should be activated by the class, if resolved.
    \return True if the entity can be added, false if it was already added.
    
    \note If the \b block is the same as the last one just added, this is
    considered an alternate selector value for the same block. However, the
    search for previously defined blocks is not extended past the last inserted
    block. This means that alternate selectors for the same block must all be
    added one after another.
    
    \note The try/catch statement will use the head symbol in the \b block SynTree
    to store the "catch/in" clause symbol.
    */   
   bool addSelectClass( Class* cls, SynTree* block );
   
   /** Adds a branch for an unknown symbol.
    \param name The name of the unkown symbol.
    \param block The block that should be activated by the class, if resolved.
    \return A requirement that should be added to the forming module.
    
    When the requirement is resolved, it must be either an integer or a class.
    The resolution process may generate an error that must be treated by
    the resolutor (the error will be returned in the resolution process).
    Duplicate names are not checked here, but they will cause a link error
    when resolved.
    
    \note If the \b block is the same as the last one just added, this is
    considered an alternate selector value for the same block. However, the
    search for previously defined blocks is not extended past the last inserted
    block. This means that alternate selectors for the same block must all be
    added one after another.
    
    \note The try/catch statement will use the head symbol in the \b block SynTree
    to store the "catch/in" clause symbol.
    */   
   Requirement* addSelectName( const String& name, SynTree* block );

   /** Finds the block that should handle this type.
    \param typeId the typeID that should be handled.
    \return A valid block handling the required type or 0 if not found.    
    */
   SynTree* findBlockForType( int64 typeId ) const;
   
   /** Finds the block that should handle this class.
    \param typeId the typeID that should be handled.
    \return A valid block handling the required class or 0 if not found.  
    
    The search for valid block is extended to base classes of cls.
    */
   SynTree* findBlockForClass( Class* cls ) const;
   
   /** Gets the block that should handle default cases.
    \return The default block or 0 if none.
    */
   SynTree* getDefault() const { return m_defaultBlock; }
   
   /** Returns a block for an item (usually the one generated by the expression).
    \param itm The item to be selected.
    \return A syntree that can handle it, or 0 if none.
    
    If the item provides a type id (either directly or through it's User class
    Class::typeID() member), the handler will be searched there; if not found,
    or if no typeID is provided, the handler will be searched in sequential
    order among the handling classes. The match will succeed as the class of
    \b itm or a base class of it is found.
    
    If both this searches fail, the default block (which may be 0) is returned.
    */
   SynTree* findBlockForItem( const Item& itm ) const;
   
   /** Sets the else branch for this if statement. 
    \param block The block that should be used to handle default choices.
    \return true if a default block was not set, false otherwise.
    
    If the function returns false, this means that another default block was
    already set, and the given \b block parameter was not used.
    
    \note The try/catch statement will use the head symbol in the \b block SynTree
    to store the "catch/in" clause symbol.
    */
   bool setDefault( SynTree* block );

   
   /** Module (used for error accounting during resolution ) */
   void module( Module* m ) { m_module = m; } 
   
   Module* module() const { return m_module; }
   
private:
   struct Private;
   Private* _p;
   
   Expression* m_expr;
   PCode m_pcExpr;
   SynTree* m_defaultBlock;
   Module* m_module;
   
   class SelectRequirement: public Requirement
   {
   public:
      SynTree* m_block;
      Class* m_cls;
      
      SelectRequirement( const String& name, SynTree* b, StmtSelect* owner ):
         Requirement( name ),
         m_block( b ),
         m_cls( 0 ),
         m_owner( owner ) 
      {}
      
      virtual ~SelectRequirement() {}
      
      virtual void onResolved( const Module* source, const Symbol* srcSym, Module* tgt, Symbol* extSym );
   
   private:
      StmtSelect* m_owner;
   };
   
   friend class SelectRequirement;
   
      
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of stmtselect.h */
