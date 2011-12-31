/*
   FALCON - The Falcon Programming Language.
   FILE: rootsyntree.cpp
   
   Root Syntactic tree.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 31 Dec 2011 20:38:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/rootsyntree.cpp"

#include <falcon/rootsyntree.h>
#include <falcon/function.h>

namespace Falcon
{

RootSynTree::RootSynTree( int line, int chr):
   SynTree( line, chr ),
   m_host(0)
{
}

RootSynTree::RootSynTree( Function* host, int line, int chr ):
   SynTree( line, chr ),
   m_host(host)
{
}

   
RootSynTree::RootSynTree( const SynTree& other ):
   SynTree(other),
   m_host(0)
{   
}

RootSynTree::~RootSynTree()
{
}

   
void RootSynTree::gcMark( uint32 mark )
{
   if( m_gcMark != mark )
   {
      m_gcMark = mark;
   
      if( m_host != 0 )
      {
         m_host->gcMark( mark );
      }
   }
}

}

/* end of rootsyntree.cpp */
