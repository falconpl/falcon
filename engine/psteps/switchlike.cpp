/*
   FALCON - The Falcon Programming Language.
   FILE: switchlike.cpp

   Parser for Falcon source files -- Switch and select base classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/switchlike.cpp"

#include <falcon/psteps/switchlike.h>
#include <falcon/syntree.h>

namespace Falcon {


SwitchlikeStatement::SwitchlikeStatement( int32 line, int32 chr ):
   Statement( line, chr ),
   m_defaultBlock(0),
   m_dummyTree(0)
{
}

SwitchlikeStatement::SwitchlikeStatement( const SwitchlikeStatement& other ):
   Statement( other ),
   m_defaultBlock(0),
   m_dummyTree(0)
{
}

SwitchlikeStatement::~SwitchlikeStatement()
{
   dispose( m_defaultBlock );
   dispose( m_dummyTree );
}


bool SwitchlikeStatement::setDefault( SynTree* block )
{
   if( ! block->setParent(this) )
   {
      return false;
   }

   dispose( m_defaultBlock );
   m_defaultBlock = block;
   return true;
}


SynTree* SwitchlikeStatement::dummyTree()
{
   if( m_dummyTree == 0 ) {
      m_dummyTree = new SynTree();
      m_dummyTree->setParent(this);
   }
   
   return m_dummyTree;
}

}

/* end of switchlike.cpp */
