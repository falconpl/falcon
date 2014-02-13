/*
   FALCON - The Falcon Programming Language.
   FILE: exprcase.h

   Syntactic tree item definitions -- Case (switch branch selector)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 16 Mar 2013 14:00:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRCASE_H
#define FALCON_EXPRCASE_H

#include <falcon/expression.h>

namespace re2 {
   class RE2;
}

namespace Falcon {
class Symbol;
class DataReader;
class DataWriter;
class CaseRequirement;
class Engine;
class ItemArray;

/** Case (switch branch selector) */
class FALCON_DYN_CLASS ExprCase: public Expression
{
public:
   ExprCase( int line =0, int chr = 0 );
   ExprCase( const ExprCase& other );
   virtual ~ExprCase();

   inline virtual ExprCase* clone() const { return new ExprCase( *this ); }
   virtual bool simplify( Item& value ) const;
   inline virtual bool isStandAlone() const {
      return false;
   }

   virtual void render( TextWriter* tw, int depth ) const;
   static void apply_( const PStep*, VMContext* ctx );

   void addNilEntry();
   void addBoolEntry( bool tof );
   void addEntry( int64 value );
   void addEntry( int64 value1, int64 value2 );
   void addEntry( const String& str1 );
   void addEntry( const String& str1, const String& str2 );
   void addEntry( re2::RE2* regex );
   void addEntry( const Symbol* symbol );
   void addEntry( Class* cls );
   bool addEntry( const Item& value );

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   virtual bool insert( int32 pos, TreeStep* element );
   virtual bool append( TreeStep* element );
   virtual bool remove( int32 pos );

   /** Changes a symbol entry in a class entry.
    *
    * Upon class resolution in select and catches statements,
    * this method gets called with the class that was found.
    */
   void onClassResolved( const String& symName, Class* cls );

   /** Verify without a context.
    * This verify version checks a "dry" entry, where symbols in the case elements
    * are not resolved.
    *
    * In case an entry is a symbol, the check returns true if the item is the same
    * symbol. This allows to check for statically declared duplicated case branches.
    */
   bool verify( const Item& item ) const;

   /** Checks if this entry is compatible with the given symbol.
    *
    * Symbols are checked literally, ignoring the value they refer to.
    */
   bool verifySymbol( const Symbol* value ) const;

   /** Verifies a live item.
    *
    * Here, symbols in the switch/select branches get
    * resolved dynamically in the context.
    */
   bool verifyItem( const Item& item, VMContext* ctx ) const;

   /** Verifies an item type.
    *
    * This verify is performed on the item type or base class.
    * This is used for select and catch branches.
    */
   bool verifyType( const Item& item, VMContext* ctx) const;

   void store( DataWriter* dw );
   void restore( DataReader* dr );

   void flatten( ItemArray& array );
   void unflatten( ItemArray& array );

private:
   class Private;
   Private* _p;

};

}

#endif

/* end of expriif.h */
