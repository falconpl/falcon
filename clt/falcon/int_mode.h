/*
   FALCON - The Falcon Programming Language.
   FILE: int_mode.h

   Falcon compiler and interpreter - interactive mode.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 23 Mar 2009 18:57:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Options storage for falcon compiler.
*/

#ifndef FALCON_INT_MODE_H
#define FALCON_INT_MODE_H

#include "falcon.h"

class AppFalcon;

class IntMode
{
   void read_line(String &line, const char* prompt);
   AppFalcon *m_owner;

public:
   IntMode( AppFalcon* owner );

   void run();
};

#endif

/* end of int_mode.h */
