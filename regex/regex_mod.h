/*
   FALCON - The Falcon Programming Language.
   FILE: regex_mod.h
   $Id: regex_mod.h,v 1.2 2007/06/23 10:14:51 jonnymind Exp $

   Module support for Regular Expressions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 18 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Module support for Regular Expressions.
*/

#ifndef flc_regex_mod_H
#define flc_regex_mod_H

#include <falcon/userdata.h>
#include <pcre.h>

#define OVECTOR_SIZE 60

namespace Falcon {

/**
   This object is being used as the carrier for the Regular Expression
   module to carry around the pre-compiled pattern.
*/

class RegexCarrier: public UserData
{
public:
   pcre *m_pattern;
   pcre_extra *m_extra;

   // vector of mathces that can be used to retreive captured strings
   int *m_ovector;
   int m_ovectorSize;
   int m_matches;

   RegexCarrier( pcre *pattern );

   virtual ~RegexCarrier();
};

}

#endif

/* end of regex_mod.h */
