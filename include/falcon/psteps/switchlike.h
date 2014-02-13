/*
   FALCON - The Falcon Programming Language.
   FILE: switchlike.h

   Parser for Falcon source files -- Switch and select base classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef _FALCON_SWITCHLIKE_H
#define _FALCON_SWITCHLIKE_H

#include <falcon/statement.h>

namespace Falcon {

class FALCON_DYN_CLASS SwitchlikeStatement: public Statement
{
public:
   SwitchlikeStatement( int32 line = 0, int32 chr = 0 );
   SwitchlikeStatement( Expression* expr, int32 line = 0, int32 chr = 0 );
   SwitchlikeStatement( const SwitchlikeStatement& other );
   virtual ~SwitchlikeStatement();

   TreeStep* selector() const;
   bool selector(TreeStep* );

   /** A dummy tree that is used during compilation to avoid unbound statements.    
    \return A temporary syntree.    
    */
   
   SynTree* dummyTree();
   
   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   virtual bool insert( int32 pos, TreeStep* element );
   virtual bool append( TreeStep* element );
   virtual bool remove( int32 pos );

   //virtual void gcMark( uint32 mark );
   SynTree* findBlockForItem( const Item& value ) const;
   SynTree* findBlockForSymbol( const Symbol* value ) const;
   SynTree* findBlockForItem( const Item& value, VMContext* ctx ) const;
   SynTree* findBlockForType( const Item& value, VMContext* ctx ) const;
   SynTree* getDefault() const;

   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual void renderHeader( TextWriter* tw, int32 depth ) const = 0;

protected:
   class Private;
   Private* _p;

   TreeStep* m_expr;
   SynTree* m_dummyTree;

private:
   template<class __TI, class __T>
   SynTree* findBlock( const __TI& item, const __T& verifier ) const;
};

}

#endif	/* _FALCON_SWITCHLIKE_H */

/* end of switchlike.h */
