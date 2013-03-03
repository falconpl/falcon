/*
   FALCON - The Falcon Programming Language.
   FILE: stmtswitch.h

   Syntactic tree item definitions -- switch statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 02 May 2012 21:18:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SWITCH_H_
#define _FALCON_SWITCH_H_

#include <falcon/psteps/switchlike.h>

namespace Falcon {

class Symbol;
class Expression;
class VMContext;

/** Handler for the switch statement.

 A branch can be added under multiple selectors.
 */
class FALCON_DYN_CLASS StmtSwitch: public SwitchlikeStatement
{
public:
   /** Create the switch statement.
    \param expr The expression that will generate the item to be selected, or 0.
    \param line The line where this statement is declared in the source.
    \param chr The character at which this statement is declared in the source.

    */
   StmtSwitch( Expression* expr = 0, int32 line = 0, int32 chr = 0 );
   StmtSwitch( const StmtSwitch& other );
   virtual ~StmtSwitch();

   virtual void describeTo( String& tgt, int depth=0 ) const;
   virtual void oneLinerTo( String& tgt ) const;
   virtual StmtSwitch* clone() const { return new StmtSwitch(*this); }

   /** Returns the selector for this expression.*/
   virtual Expression* selector() const;
   virtual bool selector( Expression* expr );
   
   /** Adds a branch for the nil type.    
    \param block The block that should be activated by that type ID.
    \return True if the entity can be added, false if the nil block is already added.    
    */
   bool addNilBlock( SynTree* block );

   /** Adds a a branch for a boolean const value.
    \param value The boolean block to be added.
    \param block The block that should be activated by that type ID.
    \return True if the entity can be added, false if the block is already added.    
    */
   bool addBoolBlock( bool value, SynTree* block );

   /** Adds a branch for an integer value.
    \param iValue The integer value for this 
    \param block The block that should be activated by that type ID.
    \return True if the entity can be added, false if the integer value
            clashes with something already added.
    
    */
   bool addIntBlock( int64 iValue, SynTree* block );
   /** Adds a branch for a range of integer values.
    \param iLow minimum value to select the branch (included).
    \param iHigh maximum value to select the branch (included).
    
    \param block The block that should be activated.
    \return True if the entity can be added, false if the value
            clashes with something already added.   
    
    \note be sure that iLow is less than or equal to iHigh before calling this
         function.
    */
   bool addRangeBlock( int64 iLow, int64 iHigh, SynTree* block );
   
   /** Adds a branch for a string selector.
    \param strLow the string that selects this branch.
    \param block The block that should be activated.
    \return True if the entity can be added, false if the value
            clashes with something already added.    
    */
   bool addStringBlock( const String& strLow, SynTree* block );
   
   /** Adds a branch for a range of strings.
    \param strLow the minimum string that selects this branch.
    \param strHigh the maximum string that selects this branch.
    \param block The block that should be activated.
    \return True if the entity can be added, false if the value
            clashes with something already added.    
    
    \note be sure that strLow is less than or equal to strHigh before calling this
         function.
    */
   bool addStringRangeBlock( const String& strLow, const String& strHigh, SynTree* block );
   
   /** Adds an arbitrary variable value to the switch.
    \param var The the variable that should be activated.
    \param block The block that should be activated.
    \return True if the block can be added, false if the variable
         symbol was already used.
    */
   bool addSymbolBlock( Symbol* var, SynTree* block );

   /** Finds the block that should handle this numeric value.
    */
   SynTree* findBlockForNumber( int64 value ) const;

   /** Finds the block that should handle this string value.
    */
   SynTree* findBlockForString( const String& value ) const;

   /** Finds the block that should handle an item.
    \param value A value to be searched in the switch block.
    \return A valid syntree or 0 if a matching block cannot be found.
    
    The item type is checked and if it's a number or a string, the switch
    integer and string values are searched.
    
    If a valid case handler is not found, 0 is returned. The default block
    should be taken in that case.
    
    This method won't check for the variable items.
    */
   SynTree* findBlockForItem( const Item& value ) const;
   
   virtual void gcMark( uint32 mark );
   
   
private:
   class Private;
   StmtSwitch::Private* _p;

   Expression* m_expr;
   SynTree* m_nilBlock;
   SynTree* m_trueBlock;
   SynTree* m_falseBlock;
   
   
   
   static void apply_( const PStep*, VMContext* ctx );
};
   
}

#endif

/* end of stmtswitch.h */
