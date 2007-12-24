/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     INTNUM = 259,
     DBLNUM = 260,
     SYMBOL = 261,
     STRING = 262,
     NIL = 263,
     END = 264,
     DEF = 265,
     WHILE = 266,
     BREAK = 267,
     CONTINUE = 268,
     DROPPING = 269,
     IF = 270,
     ELSE = 271,
     ELIF = 272,
     FOR = 273,
     FORFIRST = 274,
     FORLAST = 275,
     FORALL = 276,
     SWITCH = 277,
     CASE = 278,
     DEFAULT = 279,
     SELECT = 280,
     SENDER = 281,
     SELF = 282,
     GIVE = 283,
     TRY = 284,
     CATCH = 285,
     RAISE = 286,
     CLASS = 287,
     FROM = 288,
     OBJECT = 289,
     RETURN = 290,
     GLOBAL = 291,
     LAMBDA = 292,
     INIT = 293,
     LOAD = 294,
     LAUNCH = 295,
     CONST_KW = 296,
     ATTRIBUTES = 297,
     PASS = 298,
     EXPORT = 299,
     COLON = 300,
     FUNCDECL = 301,
     STATIC = 302,
     FORDOT = 303,
     LISTPAR = 304,
     LOOP = 305,
     OUTER_STRING = 306,
     CLOSEPAR = 307,
     OPENPAR = 308,
     CLOSESQUARE = 309,
     OPENSQUARE = 310,
     DOT = 311,
     ASSIGN_POW = 312,
     ASSIGN_SHL = 313,
     ASSIGN_SHR = 314,
     ASSIGN_BXOR = 315,
     ASSIGN_BOR = 316,
     ASSIGN_BAND = 317,
     ASSIGN_MOD = 318,
     ASSIGN_DIV = 319,
     ASSIGN_MUL = 320,
     ASSIGN_SUB = 321,
     ASSIGN_ADD = 322,
     ARROW = 323,
     FOR_STEP = 324,
     OP_TO = 325,
     COMMA = 326,
     QUESTION = 327,
     OR = 328,
     AND = 329,
     NOT = 330,
     LET = 331,
     LE = 332,
     GE = 333,
     LT = 334,
     GT = 335,
     NEQ = 336,
     EEQ = 337,
     OP_EQ = 338,
     OP_ASSIGN = 339,
     PROVIDES = 340,
     OP_NOTIN = 341,
     OP_IN = 342,
     HASNT = 343,
     HAS = 344,
     DIESIS = 345,
     ATSIGN = 346,
     CAP = 347,
     VBAR = 348,
     AMPER = 349,
     MINUS = 350,
     PLUS = 351,
     PERCENT = 352,
     SLASH = 353,
     STAR = 354,
     POW = 355,
     SHR = 356,
     SHL = 357,
     BANG = 358,
     NEG = 359,
     DECREMENT = 360,
     INCREMENT = 361,
     DOLLAR = 362
   };
#endif
/* Tokens.  */
#define EOL 258
#define INTNUM 259
#define DBLNUM 260
#define SYMBOL 261
#define STRING 262
#define NIL 263
#define END 264
#define DEF 265
#define WHILE 266
#define BREAK 267
#define CONTINUE 268
#define DROPPING 269
#define IF 270
#define ELSE 271
#define ELIF 272
#define FOR 273
#define FORFIRST 274
#define FORLAST 275
#define FORALL 276
#define SWITCH 277
#define CASE 278
#define DEFAULT 279
#define SELECT 280
#define SENDER 281
#define SELF 282
#define GIVE 283
#define TRY 284
#define CATCH 285
#define RAISE 286
#define CLASS 287
#define FROM 288
#define OBJECT 289
#define RETURN 290
#define GLOBAL 291
#define LAMBDA 292
#define INIT 293
#define LOAD 294
#define LAUNCH 295
#define CONST_KW 296
#define ATTRIBUTES 297
#define PASS 298
#define EXPORT 299
#define COLON 300
#define FUNCDECL 301
#define STATIC 302
#define FORDOT 303
#define LISTPAR 304
#define LOOP 305
#define OUTER_STRING 306
#define CLOSEPAR 307
#define OPENPAR 308
#define CLOSESQUARE 309
#define OPENSQUARE 310
#define DOT 311
#define ASSIGN_POW 312
#define ASSIGN_SHL 313
#define ASSIGN_SHR 314
#define ASSIGN_BXOR 315
#define ASSIGN_BOR 316
#define ASSIGN_BAND 317
#define ASSIGN_MOD 318
#define ASSIGN_DIV 319
#define ASSIGN_MUL 320
#define ASSIGN_SUB 321
#define ASSIGN_ADD 322
#define ARROW 323
#define FOR_STEP 324
#define OP_TO 325
#define COMMA 326
#define QUESTION 327
#define OR 328
#define AND 329
#define NOT 330
#define LET 331
#define LE 332
#define GE 333
#define LT 334
#define GT 335
#define NEQ 336
#define EEQ 337
#define OP_EQ 338
#define OP_ASSIGN 339
#define PROVIDES 340
#define OP_NOTIN 341
#define OP_IN 342
#define HASNT 343
#define HAS 344
#define DIESIS 345
#define ATSIGN 346
#define CAP 347
#define VBAR 348
#define AMPER 349
#define MINUS 350
#define PLUS 351
#define PERCENT 352
#define SLASH 353
#define STAR 354
#define POW 355
#define SHR 356
#define SHL 357
#define BANG 358
#define NEG 359
#define DECREMENT 360
#define INCREMENT 361
#define DOLLAR 362




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union
#line 67 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
lex_value_t {
   Falcon::int64 integer;
   Falcon::numeric numeric;
   char * charp;
   Falcon::String *stringp;
   Falcon::Symbol *symbol;
   Falcon::Value *fal_val;
   Falcon::Expression *fal_exp;
   Falcon::Statement *fal_stat;
   Falcon::ArrayDecl *fal_adecl;
   Falcon::DictDecl *fal_ddecl;
   Falcon::SymbolList *fal_symlist;
}
/* Line 1489 of yacc.c.  */
#line 277 "/home/gian/Progetti/falcon/core/engine/src_parser.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



