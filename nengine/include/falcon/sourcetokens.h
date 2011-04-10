/*
   FALCON - The Falcon Programming Language.
   FILE: sourcetokens.h

   Definition of grammar tokens known by the Falcon source parser.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 10 Apr 2011 23:13:02 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef _FALCON_SOURCETOKENS_H
#define	_FALCON_SOURCETOKENS_H


#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/parser/terminal.h>

namespace Falcon {

extern FALCON_DYN_SYM Parsing::Terminal t_dot;

extern FALCON_DYN_SYM Parsing::Terminal t_openpar;
extern FALCON_DYN_SYM Parsing::Terminal t_closepar;
extern FALCON_DYN_SYM Parsing::Terminal t_opensquare;
extern FALCON_DYN_SYM Parsing::Terminal t_closesquare;
extern FALCON_DYN_SYM Parsing::Terminal t_opengraph;
extern FALCON_DYN_SYM Parsing::Terminal t_closegraph;

extern FALCON_DYN_SYM Parsing::Terminal t_plus;
extern FALCON_DYN_SYM Parsing::Terminal t_minus;
extern FALCON_DYN_SYM Parsing::Terminal t_times;
extern FALCON_DYN_SYM Parsing::Terminal t_divide;
extern FALCON_DYN_SYM Parsing::Terminal t_modulo;
extern FALCON_DYN_SYM Parsing::Terminal t_pow;

extern FALCON_DYN_SYM Parsing::Terminal t_token_not;


extern FALCON_DYN_SYM Parsing::Terminal t_token_as;
extern FALCON_DYN_SYM Parsing::Terminal t_token_eq;
extern FALCON_DYN_SYM Parsing::Terminal t_token_if;
extern FALCON_DYN_SYM Parsing::Terminal t_token_in;
extern FALCON_DYN_SYM Parsing::Terminal t_token_or;
extern FALCON_DYN_SYM Parsing::Terminal t_token_to;

extern FALCON_DYN_SYM Parsing::Terminal t_token_not;
extern FALCON_DYN_SYM Parsing::Terminal t_token_end;
extern FALCON_DYN_SYM Parsing::Terminal t_token_nil;

}

#endif	/* _FALCON_SOURCETOKENS_H */

/* end of sourcetokens.h */
