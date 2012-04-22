/*
   FALCON - The Falcon Programming Language.
   FILE: regex_mod.h

   Module support for Regular Expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 18 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Module support for Regular Expressions.
*/

#ifndef flc_regex_mod_H
#define flc_regex_mod_H

#include <falcon/types.h>
#include <pcre.h>

#define OVECTOR_SIZE 60

namespace Falcon {

/**
   This object is being used as the carrier for the Regular Expression
   module to carry around the pre-compiled pattern.
*/

class RegexCarrier
{
public:
   pcre *m_pattern;
   pcre_extra *m_extra;

   // vector of mathces that can be used to retreive captured strings
   int *m_ovector;
   int m_ovectorSize;
   int m_matches;

   uint32 m_mark;

   RegexCarrier();
   void init( pcre *pattern );

   virtual ~RegexCarrier();
};

}

#endif

/* end of regex_mod.h */
