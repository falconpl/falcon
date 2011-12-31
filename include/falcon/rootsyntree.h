/*
   FALCON - The Falcon Programming Language.
   FILE: rootsyntree.h

   Root Syntactic tree.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 31 Dec 2011 20:38:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_ROOTSYNTREE_H
#define FALCON_ROOTSYNTREE_H

#include <falcon/setup.h>
#include <falcon/syntree.h>

namespace Falcon
{

class Function;

/** Root Syntactic tree.
 This is just a SynTree that knows being at root of a statement tree; it is used
 as a component of Syntactic Functions and propagates any mark it receives to
 its host.
 */
class FALCON_DYN_CLASS RootSynTree: public SynTree
{

public:
   RootSynTree( int line = 0, int chr = 0);
   RootSynTree( Function* host, int line = 0, int chr = 0);
   RootSynTree( const SynTree& other );
   virtual ~RootSynTree();

   Function* host() const { return m_host; }
   void host( Function* h ) { m_host = h; }
   
   virtual void gcMark( uint32 mark );
   
protected:   
   Function* m_host;
};

}

#endif

/* end of rootsyntree.h */

