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
     NAME = 259,
     COMMA = 260,
     COLON = 261,
     DENTRY = 262,
     DGLOBAL = 263,
     DVAR = 264,
     DCONST = 265,
     DATTRIB = 266,
     DLOCAL = 267,
     DPARAM = 268,
     DFUNCDEF = 269,
     DENDFUNC = 270,
     DFUNC = 271,
     DMETHOD = 272,
     DPROP = 273,
     DPROPREF = 274,
     DCLASS = 275,
     DCLASSDEF = 276,
     DCTOR = 277,
     DENDCLASS = 278,
     DFROM = 279,
     DEXTERN = 280,
     DMODULE = 281,
     DLOAD = 282,
     DSWITCH = 283,
     DSELECT = 284,
     DCASE = 285,
     DENDSWITCH = 286,
     DLINE = 287,
     DSTRING = 288,
     DCSTRING = 289,
     DHAS = 290,
     DHASNT = 291,
     DINHERIT = 292,
     DINSTANCE = 293,
     SYMBOL = 294,
     EXPORT = 295,
     LABEL = 296,
     INTEGER = 297,
     REG_A = 298,
     REG_B = 299,
     REG_S1 = 300,
     REG_S2 = 301,
     NUMERIC = 302,
     STRING = 303,
     STRING_ID = 304,
     I_LD = 305,
     I_LNIL = 306,
     NIL = 307,
     I_ADD = 308,
     I_SUB = 309,
     I_MUL = 310,
     I_DIV = 311,
     I_MOD = 312,
     I_POW = 313,
     I_ADDS = 314,
     I_SUBS = 315,
     I_MULS = 316,
     I_DIVS = 317,
     I_POWS = 318,
     I_INC = 319,
     I_DEC = 320,
     I_NEG = 321,
     I_NOT = 322,
     I_RET = 323,
     I_RETV = 324,
     I_RETA = 325,
     I_FORK = 326,
     I_PUSH = 327,
     I_PSHR = 328,
     I_PSHN = 329,
     I_POP = 330,
     I_LDV = 331,
     I_LDVT = 332,
     I_STV = 333,
     I_STVR = 334,
     I_STVS = 335,
     I_LDP = 336,
     I_LDPT = 337,
     I_STP = 338,
     I_STPR = 339,
     I_STPS = 340,
     I_TRAV = 341,
     I_TRAN = 342,
     I_TRAL = 343,
     I_IPOP = 344,
     I_XPOP = 345,
     I_GENA = 346,
     I_GEND = 347,
     I_GENR = 348,
     I_GEOR = 349,
     I_JMP = 350,
     I_IFT = 351,
     I_IFF = 352,
     I_BOOL = 353,
     I_EQ = 354,
     I_NEQ = 355,
     I_GT = 356,
     I_GE = 357,
     I_LT = 358,
     I_LE = 359,
     I_UNPK = 360,
     I_UNPS = 361,
     I_CALL = 362,
     I_INST = 363,
     I_SWCH = 364,
     I_SELE = 365,
     I_NOP = 366,
     I_TRY = 367,
     I_JTRY = 368,
     I_PTRY = 369,
     I_RIS = 370,
     I_LDRF = 371,
     I_ONCE = 372,
     I_BAND = 373,
     I_BOR = 374,
     I_BXOR = 375,
     I_BNOT = 376,
     I_MODS = 377,
     I_AND = 378,
     I_OR = 379,
     I_ANDS = 380,
     I_ORS = 381,
     I_XORS = 382,
     I_NOTS = 383,
     I_HAS = 384,
     I_HASN = 385,
     I_GIVE = 386,
     I_GIVN = 387,
     I_IN = 388,
     I_NOIN = 389,
     I_PROV = 390,
     I_END = 391,
     I_PEEK = 392,
     I_PSIN = 393,
     I_PASS = 394,
     I_FORI = 395,
     I_FORN = 396,
     I_SHR = 397,
     I_SHL = 398,
     I_SHRS = 399,
     I_SHLS = 400,
     I_LDVR = 401,
     I_LDPR = 402,
     I_LSB = 403,
     I_INDI = 404,
     I_STEX = 405,
     I_TRAC = 406,
     I_WRT = 407
   };
#endif
/* Tokens.  */
#define EOL 258
#define NAME 259
#define COMMA 260
#define COLON 261
#define DENTRY 262
#define DGLOBAL 263
#define DVAR 264
#define DCONST 265
#define DATTRIB 266
#define DLOCAL 267
#define DPARAM 268
#define DFUNCDEF 269
#define DENDFUNC 270
#define DFUNC 271
#define DMETHOD 272
#define DPROP 273
#define DPROPREF 274
#define DCLASS 275
#define DCLASSDEF 276
#define DCTOR 277
#define DENDCLASS 278
#define DFROM 279
#define DEXTERN 280
#define DMODULE 281
#define DLOAD 282
#define DSWITCH 283
#define DSELECT 284
#define DCASE 285
#define DENDSWITCH 286
#define DLINE 287
#define DSTRING 288
#define DCSTRING 289
#define DHAS 290
#define DHASNT 291
#define DINHERIT 292
#define DINSTANCE 293
#define SYMBOL 294
#define EXPORT 295
#define LABEL 296
#define INTEGER 297
#define REG_A 298
#define REG_B 299
#define REG_S1 300
#define REG_S2 301
#define NUMERIC 302
#define STRING 303
#define STRING_ID 304
#define I_LD 305
#define I_LNIL 306
#define NIL 307
#define I_ADD 308
#define I_SUB 309
#define I_MUL 310
#define I_DIV 311
#define I_MOD 312
#define I_POW 313
#define I_ADDS 314
#define I_SUBS 315
#define I_MULS 316
#define I_DIVS 317
#define I_POWS 318
#define I_INC 319
#define I_DEC 320
#define I_NEG 321
#define I_NOT 322
#define I_RET 323
#define I_RETV 324
#define I_RETA 325
#define I_FORK 326
#define I_PUSH 327
#define I_PSHR 328
#define I_PSHN 329
#define I_POP 330
#define I_LDV 331
#define I_LDVT 332
#define I_STV 333
#define I_STVR 334
#define I_STVS 335
#define I_LDP 336
#define I_LDPT 337
#define I_STP 338
#define I_STPR 339
#define I_STPS 340
#define I_TRAV 341
#define I_TRAN 342
#define I_TRAL 343
#define I_IPOP 344
#define I_XPOP 345
#define I_GENA 346
#define I_GEND 347
#define I_GENR 348
#define I_GEOR 349
#define I_JMP 350
#define I_IFT 351
#define I_IFF 352
#define I_BOOL 353
#define I_EQ 354
#define I_NEQ 355
#define I_GT 356
#define I_GE 357
#define I_LT 358
#define I_LE 359
#define I_UNPK 360
#define I_UNPS 361
#define I_CALL 362
#define I_INST 363
#define I_SWCH 364
#define I_SELE 365
#define I_NOP 366
#define I_TRY 367
#define I_JTRY 368
#define I_PTRY 369
#define I_RIS 370
#define I_LDRF 371
#define I_ONCE 372
#define I_BAND 373
#define I_BOR 374
#define I_BXOR 375
#define I_BNOT 376
#define I_MODS 377
#define I_AND 378
#define I_OR 379
#define I_ANDS 380
#define I_ORS 381
#define I_XORS 382
#define I_NOTS 383
#define I_HAS 384
#define I_HASN 385
#define I_GIVE 386
#define I_GIVN 387
#define I_IN 388
#define I_NOIN 389
#define I_PROV 390
#define I_END 391
#define I_PEEK 392
#define I_PSIN 393
#define I_PASS 394
#define I_FORI 395
#define I_FORN 396
#define I_SHR 397
#define I_SHL 398
#define I_SHRS 399
#define I_SHLS 400
#define I_LDVR 401
#define I_LDPR 402
#define I_LSB 403
#define I_INDI 404
#define I_STEX 405
#define I_TRAC 406
#define I_WRT 407




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



