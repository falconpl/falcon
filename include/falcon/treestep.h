/*
   FALCON - The Falcon Programming Language.
   FILE: treestep.h

   PStep that can be inserted in a code tree.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Dec 2011 14:38:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_TREESTEP_H
#define FALCON_TREESTEP_H

#include <falcon/pstep.h>

namespace Falcon {

class Class;
class Expression;
class Statement;
class SynTree;
class Item;

/** PStep extension that can be insereted in an exposable syntactic tree.
 
 The basic PStep is just an istruction for the Falcon organic virtual machine.
 This extensions has the basic traits needed to support source code representation
 as PStep instances, and to expose a PStep reflexively to
 a Falcon program, or serialize it in pre-compiled Falcon module files.
 
 In short, all the expressions, statements and syntactic trees must derive from
 this class.
 
 Each class in this hierarcy has a Class handler that is used both to serialize
 and to handle the step at script level.
 
 @section treestep_tree_interface TreeStep tree interface.
 
 Each TreeStep represents an atomic grammar entity that can be written in a 
 source program usually (but not necessarily) in Falcon langauge. As such, it
 offers an interface that allows to create coherent source code trees.
 
 The interface is composed of a set of virtual functions that each concrete
 subclass must reimplement to offer a common behavior:
 
 - arity(): the size of the elements in the source tree code below this one. 
   Can be 0.
 - nth( int ): Returns the nth element in 0..arity(). Negative values return the
   nth-last element.
 - nth( int, TreeStep* ): sets the nth element (see concerns about parenting and
 ownership). It is legal to set a step as 0 (i.e. for optional blocks as firfirst in for/in).
 - insert(int, TreeStep*): Inserts a TreeStep at given position.
 - remove( int ): Removes a TreeStep at given position.
 
 Not all the TreeSteps can fulfill every request. Some TreeStep won't accept a
 0 element, others have fixed arity and won't accept insert/remove requests. In that
 cases, the method just need to return false and deny performing the required
 operation.
 
 @section treestep_relationship_rules TreeStep akin relationships.
 
 Tree Steps are organized into three categories:
 - Syntactic trees: they represent an unitary block of code, typically stored
   under a statement or composing the body of a function. The code block is
   composed of a list of statements, and can have a target symbol and a selector
   expression.
 - Statement: Toplevel script instruction having a complete semantic meaning.
   it is typically, but not necessarily, composed of one or more blocks ( each one
   is a syntactic tree).
 - Expression: An expression is a set of an operation and zero or more operands,
   each one must be an expression.
 
 Typically, the vast majority of expressions, statements and syntactic trees
 respect this structure, although some present some peculiarities. For instance,
 the ExprSymbol, reperenting access to a symbol, has a target symbol, while
 the return statement has not any block, but it has a selector expression.
 
 The three TreeStep categories differ for the way they interact with the virtual
 machine. 
 
 An Expression will \b invariantly consume a number of data from the data stack
 equal to its own arity, and push \b exactly one result. 
 
 A Statement is bound to respect the data stack size; once it abandons the virtual
 machine (through the usual VMContext::popCode method) it must leave on the
 data stack the same amount of data it did find when it was first called. Usually,
 it is just necessary to undo what the expressions used by the statement have
 done, which usually means to pop a single value left by the selector expression.
 
 SynTrees coordinate the work of multiple statements, cooperating with some
 special statement (as the autoexpression) to pass data back to the virtual machine
 user (i.e. an interactive mode which needs to know the result of the last
 operation), or controlling automatically the flow of the code as in the case
 of the RuleSynTree.
 
 TreeStep::insert() and TreeStep::nth() operations will allow only certain kind
 of TreeStep to be added to the Step. SynTree only accept Statement instances,
 Statement instances only accept SynTree instances and Expression only accepts
 expressions:
 
 - SynTree --> hosts Statement
 - Statement --> hosts SynTree
 - Expression --> hosts Expression
 
 @note This rule is expressed by the TreeStep::canHost() method.
 
 Keep in mind that arity(), nth(), insert() and remove() are proxy functions
 that are re-implemented by each concrete class to perform a coherent behavior.
 For instance, the While staement has just one syntree block, which can be
 directly accessed through a proper (inlined) method in the While statement.
 
 The reason for the general interface accessing sub-trees is that to minimize
 the code needed to specifically manage access to the hierarchical formation
 of the code structure, and to serialize/deserialize it. Thanks to this interface,
 only a handful of specialized PSteps need 
 
 @section treestep_ownership_rules TreeStep ownership and marking.
 // TODO Write
 
 @section treestep_writing_guiide TreeStep subclassing guidelines.
 
 TreeSteps must follow certain rules so that they can participate in the 
 serialization, cloning and memory management protocols that involve the
 representation of language tokens at syntactic level.
 
 All the TreeStep subclasses must:
 - Provide an empty constructor -- minimally accepting the SourceDef parameters
   (line and character) as default zero.
 - Provide a copy constructor and a virtual clone() method returning the same
   type of the subclass. The cloning process must ensure that the parent of 
   the cloned entity is 0.
 - Declare their Class handler in all the constructors through the expression:
   @code
      m_class = classInstance;
   @endcode
    Notice that standard language classes use the macro FALCON_DECLARE_SYN_CLASS( syntoken ),
    declared in synclasses.h, where all the handlers for standard language classes are hosted.
 - Check for "blanking" in describeTo and eventually oneLiner ovverrides. Blank entities
   are those created through the empty constructor, and are not ready to be used.
   The representation must be "&gt;Blank classname&lt;"
 - Assert via fassert for all the necessary elements of the entity to have been filled
   in "apply". Debug check is enough, as an unprepared Step should not reach execution
   in release mode once the code is tested in debug.
 - provide arity(), nth(), insert(), remove() and selector() overrides as sensible.
   those overrides are exposed as a part of every TreeStep subclass to the scripts,
   and arity() and nth() are used by the default ClassTreeStep::flatten() and 
   ClassTreeStep::unflatten() during serialization.
 - Correctly parenting all the TreeStep that are set as belonging to the parent,
   and refuse to accept entities that have already a parent.
 
 Mainly, TreeStep subclasses should inherit from Statement, Expression or
 SynTree, which also define one of the main categories of tree step.
 */
class FALCON_DYN_CLASS TreeStep: public PStep
{
public:
   inline TreeStep( const TreeStep& other ):
      PStep( other ),
      m_handler( other.m_handler ),
      m_cat( other.m_cat ),
      m_parent( 0 )
   {}
      
   virtual ~TreeStep() {}
   
   typedef enum {
      e_cat_statement,
      e_cat_syntree,
      e_cat_expression
   }
   t_category; 
   
   t_category category() const { return m_cat; }
   
   /** Gets the handler class of this step.
    \return a valid handler class or 0 if the step cannot be handled.
    
    A handler class is used by the VM and by the serialization process to
    expose a PStep to a script. Some PSteps are internally used by the VM and
    are not visible as static grammar entities, so it's perfectly legit to
    have PSteps without a handler class.
    
    \note the class is not owned nor 
    */
   Class* handler() const { return m_handler; }
   
   /** Sets the handler of this step.
   \param cls The handler Class of this step.
    */
   void handler( Class* cls ) { m_handler = cls; }
   
   /** Marks this TreeStep, its parent or its function.
    \param mark The GC mark    
    This method is called by the class handler ClassTreeStep
    that marks exclusively the topmost TreeStep in a hierarcy.
    

    */
   virtual void gcMark( uint32 mark );
   
   /** Returns the current mark for this step */
   uint32 gcMark() const { return m_gcMark; }
   
   /** Sets the parent for this tree step. 
    \param ts The TreeStep to be assumed as parent. 
    \return true if the parent can be set, false if it was already set.
        
    */
   bool setParent( TreeStep* ts );
   
   /** Returns the parent of this TreeStep.
    \return a valid TreeStep or 0 if this TreeStep has no parent. 
    */
   TreeStep* parent() const { return m_parent; }
   
   /** Clone this tree step.
    All the steps in a tree must be cloneable.    
    */
   virtual TreeStep* clone() const = 0;
   
   /** Checks if the given item is a TreeStep of type expression.
    \param item The item to be checked.
    \param bCreate true to allow creation of ExprSymbol or ExprValue for non-syntree
    expressions. On exit, will be true if the expression as been created anew.
    \return 0 If the item is not an expression, a valid expression otherwise .
    */
   static Expression* checkExpr( const Item& item, bool& bCreate );
   
   /** Checks if the given item is a TreeStep of type statement.
    \param item The item to be checked.
    \return 0 If the item is not a statement, a valid statement otherwise .
    */
   static Statement* checkStatement( const Item& item );
   
   /** Checks if the given item is a TreeStep of type syntree.
    \param item The item to be checked.
    \return 0 If the item is not a syntree, a valid syntree otherwise .
    */
   static SynTree* checkSyntree( const Item& item );
      
   //=================================================
   // Arity
   //
   /** Arity of the tree step. 
    This indicates how many sub-elements this element has.
        
    */
   virtual int32 arity() const;
   
   /** Nth sub-element of this element in 0..arity() */
   virtual TreeStep* nth( int32 n ) const;
   
   /** Setting the nth sub-element.
    \param n The sub-element number.
    \param ts An unparented expression.
    \return true if \b ts can be parented and n is valid, false otherwise.
    
    If a previous expression occupies this position, it is destroyed.    
    */
   virtual bool setNth( int32 n, TreeStep* ts );
   
   /** Inserts a sub-element in this element.
    \param pos The position BEFORE which to insert the new expression.
    \param element An unparented expression.
    \return true if \b element can be parented inserted, false otherwise.
    
    If this expression allows varying arity, and the received \b element
    can be set, the method returns true, otherwise it returns false.
    
    The default behavior is that of returning false.
    
    The position can be negative; in that case, it's calculated from end
    (-1 being the last element).
    
    If the position is out of range, the expression is inserted at end (appended).
    */
   virtual bool insert( int32 pos, TreeStep* element );
   
   /** Removes a sub-element in this element.
    \param pos The position of the expression to be removed.
    \return true if \b expr can be parented removed, false otherwise.
    
    If this expression allows varying arity, the expression at the given
    position is removed.
    
    The default behavior is that of returning false.
    
    The position can be negative; in that case, it's calculated from end
    (-1 being the last element).
    
    If the position is out of range, the method returns false.
    */
   virtual bool remove( int32 pos );
   
   /** Returns the selector of this tree step.
    \return the selector of this step or 0.
    
    Many tree elements have a "selector", that is, a single expression that
    has a relevant meaning (ususally selects if a block or statement must be
    entered or not). Selectors are always expressions, and various elements
    can expose them. SynTrees always have it (even if not all the syntrees will
    use it), many statements have and even some expressions (as the fast-if
    expression).
    
    If the step has not a selector, this method returns 0.
    */
   virtual Expression* selector() const; 
   
   /** Returns the selector of this tree step.
    \param e An expression that will be used as a selector.
    \return true if this step accepts a selector, false otherwise.
    
    */
   virtual bool selector( Expression* e ); 
   
   /** Check if this TreeStep could host the given target.
    \param target A PStep that might be inserted here.
    \return True if the category of the given target is compatible with objects
      owned by this step
   */
   bool canHost( TreeStep* target ) { return canHost( target->category() ); }
   
   /** Check if this TreeStep could host the given target category.
    \param cat A category that might be inserted here.
    \return True if the category is compatible with objects
      owned by this step.
   */
   bool canHost( t_category cat ) {
      return (m_cat == e_cat_statement && cat == e_cat_syntree) 
         || (m_cat == e_cat_syntree && cat == e_cat_statement)
         || (m_cat == e_cat_expression && cat == e_cat_expression);
   }
   
   /** Searches for unquoted expressions in literal statements. 
    \param sender The treestep requiring unquotes to declare themselves.
    
    The default behavior is that of invoking registerUnquote on all the
    arity-visible children and on the selectors.
    
    Unquoted expressions should call subscribeUnquote on the sender.
    */
   virtual void registerUnquotes( TreeStep* sender ); 
   
   /** Subscribes an unquoted expression to the sender.
    \param unquote The unquote expression being registered.
    
    The base method does nothing.
    */
   virtual void subscribeUnquote( Expression* unquote ); 
   
protected:
   uint32 m_gcMark; 
   Class* m_handler;
   t_category m_cat;
   TreeStep* m_parent;
   
   TreeStep( Class* cls, t_category t, int line = 0, int chr = 0 ):
      PStep(line, chr ),
      m_handler( cls ),
      m_cat(t),
      m_parent(0)
   {}
   
   // Unclassesd step, or class provided later on
   TreeStep( t_category t, int line = 0, int chr = 0 ):
      PStep(line, chr ),
      m_handler( 0 ),
      m_cat(t),
      m_parent(0)
   {}

};

}

#endif	/* FALCON_TREESTEP_H */

/* end of treestep.h */
