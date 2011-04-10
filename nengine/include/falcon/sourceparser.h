/* 
 * File:   sourceparser.h
 * Author: gian
 *
 * Created on 11 aprile 2011, 0.40
 */

#ifndef SOURCEPARSER_H
#define	SOURCEPARSER_H

/*
   FALCON - The Falcon Programming Language.
   FILE: sourceparser.cpp

   Token for the parser subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Apr 2011 00:04:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/parser/parser.h>

namespace Falcon {
class SynTree;

/** Class reading a Falcon script source.
 */
class FALCON_DYN_CLASS SourceParser: public Parsing::Parser
{
public:
   SourceParser( SynTree* st );
   bool parse();
   
private:
   SynTree* m_syntree;
};

}

#endif	/* SOURCEPARSER_H */

/* end of sourceparser.h */
