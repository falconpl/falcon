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
     REG_L1 = 302,
     REG_L2 = 303,
     NUMERIC = 304,
     STRING = 305,
     STRING_ID = 306,
     TRUE_TOKEN = 307,
     FALSE_TOKEN = 308,
     I_LD = 309,
     I_LNIL = 310,
     NIL = 311,
     I_ADD = 312,
     I_SUB = 313,
     I_MUL = 314,
     I_DIV = 315,
     I_MOD = 316,
     I_POW = 317,
     I_ADDS = 318,
     I_SUBS = 319,
     I_MULS = 320,
     I_DIVS = 321,
     I_POWS = 322,
     I_INC = 323,
     I_DEC = 324,
     I_INCP = 325,
     I_DECP = 326,
     I_NEG = 327,
     I_NOT = 328,
     I_RET = 329,
     I_RETV = 330,
     I_RETA = 331,
     I_FORK = 332,
     I_PUSH = 333,
     I_PSHR = 334,
     I_PSHN = 335,
     I_POP = 336,
     I_LDV = 337,
     I_LDVT = 338,
     I_STV = 339,
     I_STVR = 340,
     I_STVS = 341,
     I_LDP = 342,
     I_LDPT = 343,
     I_STP = 344,
     I_STPR = 345,
     I_STPS = 346,
     I_TRAV = 347,
     I_TRAN = 348,
     I_TRAL = 349,
     I_IPOP = 350,
     I_XPOP = 351,
     I_GENA = 352,
     I_GEND = 353,
     I_GENR = 354,
     I_GEOR = 355,
     I_JMP = 356,
     I_IFT = 357,
     I_IFF = 358,
     I_BOOL = 359,
     I_EQ = 360,
     I_NEQ = 361,
     I_GT = 362,
     I_GE = 363,
     I_LT = 364,
     I_LE = 365,
     I_UNPK = 366,
     I_UNPS = 367,
     I_CALL = 368,
     I_INST = 369,
     I_SWCH = 370,
     I_SELE = 371,
     I_NOP = 372,
     I_TRY = 373,
     I_JTRY = 374,
     I_PTRY = 375,
     I_RIS = 376,
     I_LDRF = 377,
     I_ONCE = 378,
     I_BAND = 379,
     I_BOR = 380,
     I_BXOR = 381,
     I_BNOT = 382,
     I_MODS = 383,
     I_AND = 384,
     I_OR = 385,
     I_ANDS = 386,
     I_ORS = 387,
     I_XORS = 388,
     I_NOTS = 389,
     I_HAS = 390,
     I_HASN = 391,
     I_GIVE = 392,
     I_GIVN = 393,
     I_IN = 394,
     I_NOIN = 395,
     I_PROV = 396,
     I_END = 397,
     I_PEEK = 398,
     I_PSIN = 399,
     I_PASS = 400,
     I_SHR = 401,
     I_SHL = 402,
     I_SHRS = 403,
     I_SHLS = 404,
     I_LDVR = 405,
     I_LDPR = 406,
     I_LSB = 407,
     I_INDI = 408,
     I_STEX = 409,
     I_TRAC = 410,
     I_WRT = 411,
     I_STO = 412
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
#define REG_L1 302
#define REG_L2 303
#define NUMERIC 304
#define STRING 305
#define STRING_ID 306
#define TRUE_TOKEN 307
#define FALSE_TOKEN 308
#define I_LD 309
#define I_LNIL 310
#define NIL 311
#define I_ADD 312
#define I_SUB 313
#define I_MUL 314
#define I_DIV 315
#define I_MOD 316
#define I_POW 317
#define I_ADDS 318
#define I_SUBS 319
#define I_MULS 320
#define I_DIVS 321
#define I_POWS 322
#define I_INC 323
#define I_DEC 324
#define I_INCP 325
#define I_DECP 326
#define I_NEG 327
#define I_NOT 328
#define I_RET 329
#define I_RETV 330
#define I_RETA 331
#define I_FORK 332
#define I_PUSH 333
#define I_PSHR 334
#define I_PSHN 335
#define I_POP 336
#define I_LDV 337
#define I_LDVT 338
#define I_STV 339
#define I_STVR 340
#define I_STVS 341
#define I_LDP 342
#define I_LDPT 343
#define I_STP 344
#define I_STPR 345
#define I_STPS 346
#define I_TRAV 347
#define I_TRAN 348
#define I_TRAL 349
#define I_IPOP 350
#define I_XPOP 351
#define I_GENA 352
#define I_GEND 353
#define I_GENR 354
#define I_GEOR 355
#define I_JMP 356
#define I_IFT 357
#define I_IFF 358
#define I_BOOL 359
#define I_EQ 360
#define I_NEQ 361
#define I_GT 362
#define I_GE 363
#define I_LT 364
#define I_LE 365
#define I_UNPK 366
#define I_UNPS 367
#define I_CALL 368
#define I_INST 369
#define I_SWCH 370
#define I_SELE 371
#define I_NOP 372
#define I_TRY 373
#define I_JTRY 374
#define I_PTRY 375
#define I_RIS 376
#define I_LDRF 377
#define I_ONCE 378
#define I_BAND 379
#define I_BOR 380
#define I_BXOR 381
#define I_BNOT 382
#define I_MODS 383
#define I_AND 384
#define I_OR 385
#define I_ANDS 386
#define I_ORS 387
#define I_XORS 388
#define I_NOTS 389
#define I_HAS 390
#define I_HASN 391
#define I_GIVE 392
#define I_GIVN 393
#define I_IN 394
#define I_NOIN 395
#define I_PROV 396
#define I_END 397
#define I_PEEK 398
#define I_PSIN 399
#define I_PASS 400
#define I_SHR 401
#define I_SHL 402
#define I_SHRS 403
#define I_SHLS 404
#define I_LDVR 405
#define I_LDPR 406
#define I_LSB 407
#define I_INDI 408
#define I_STEX 409
#define I_TRAC 410
#define I_WRT 411
#define I_STO 412




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



