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
   SwitchlikeStatement( const SwitchlikeStatement& other );
   virtual ~SwitchlikeStatement();

   /** A dummy tree that is used during compilation to avoid unbound statements.    
    \return A temporary syntree.    
    */
   
   SynTree* dummyTree();
   
   
   /** Gets the block that should handle default cases.
    \return The default block or 0 if none.
    */
   SynTree* getDefault() const { return m_defaultBlock; }


   /** Sets the else branch for this if statement.
    \param block The block that should be used to handle default choices.
    \return true if a default block was not set, false otherwise.

    If the function returns false, this means that another default block was
    already set, and the given \b block parameter was not used.  
    */
   bool setDefault( SynTree* block );
   
protected:
   SynTree* m_defaultBlock;   
   SynTree* m_dummyTree;   
};

}

#endif	/* _FALCON_SWITCHLIKE_H */

/* end of switchlike.h */
