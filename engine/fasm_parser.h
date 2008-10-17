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
     DIMPORT = 283,
     DALIAS = 284,
     DSWITCH = 285,
     DSELECT = 286,
     DCASE = 287,
     DENDSWITCH = 288,
     DLINE = 289,
     DSTRING = 290,
     DISTRING = 291,
     DCSTRING = 292,
     DHAS = 293,
     DHASNT = 294,
     DINHERIT = 295,
     DINSTANCE = 296,
     SYMBOL = 297,
     EXPORT = 298,
     LABEL = 299,
     INTEGER = 300,
     REG_A = 301,
     REG_B = 302,
     REG_S1 = 303,
     REG_S2 = 304,
     REG_L1 = 305,
     REG_L2 = 306,
     NUMERIC = 307,
     STRING = 308,
     STRING_ID = 309,
     TRUE_TOKEN = 310,
     FALSE_TOKEN = 311,
     I_LD = 312,
     I_LNIL = 313,
     NIL = 314,
     I_ADD = 315,
     I_SUB = 316,
     I_MUL = 317,
     I_DIV = 318,
     I_MOD = 319,
     I_POW = 320,
     I_ADDS = 321,
     I_SUBS = 322,
     I_MULS = 323,
     I_DIVS = 324,
     I_POWS = 325,
     I_INC = 326,
     I_DEC = 327,
     I_INCP = 328,
     I_DECP = 329,
     I_NEG = 330,
     I_NOT = 331,
     I_RET = 332,
     I_RETV = 333,
     I_RETA = 334,
     I_FORK = 335,
     I_PUSH = 336,
     I_PSHR = 337,
     I_PSHN = 338,
     I_POP = 339,
     I_LDV = 340,
     I_LDVT = 341,
     I_STV = 342,
     I_STVR = 343,
     I_STVS = 344,
     I_LDP = 345,
     I_LDPT = 346,
     I_STP = 347,
     I_STPR = 348,
     I_STPS = 349,
     I_TRAV = 350,
     I_TRAN = 351,
     I_TRAL = 352,
     I_IPOP = 353,
     I_XPOP = 354,
     I_GENA = 355,
     I_GEND = 356,
     I_GENR = 357,
     I_GEOR = 358,
     I_JMP = 359,
     I_IFT = 360,
     I_IFF = 361,
     I_BOOL = 362,
     I_EQ = 363,
     I_NEQ = 364,
     I_GT = 365,
     I_GE = 366,
     I_LT = 367,
     I_LE = 368,
     I_UNPK = 369,
     I_UNPS = 370,
     I_CALL = 371,
     I_INST = 372,
     I_SWCH = 373,
     I_SELE = 374,
     I_NOP = 375,
     I_TRY = 376,
     I_JTRY = 377,
     I_PTRY = 378,
     I_RIS = 379,
     I_LDRF = 380,
     I_ONCE = 381,
     I_BAND = 382,
     I_BOR = 383,
     I_BXOR = 384,
     I_BNOT = 385,
     I_MODS = 386,
     I_AND = 387,
     I_OR = 388,
     I_ANDS = 389,
     I_ORS = 390,
     I_XORS = 391,
     I_NOTS = 392,
     I_HAS = 393,
     I_HASN = 394,
     I_GIVE = 395,
     I_GIVN = 396,
     I_IN = 397,
     I_NOIN = 398,
     I_PROV = 399,
     I_END = 400,
     I_PEEK = 401,
     I_PSIN = 402,
     I_PASS = 403,
     I_SHR = 404,
     I_SHL = 405,
     I_SHRS = 406,
     I_SHLS = 407,
     I_LDVR = 408,
     I_LDPR = 409,
     I_LSB = 410,
     I_INDI = 411,
     I_STEX = 412,
     I_TRAC = 413,
     I_WRT = 414,
     I_STO = 415,
     I_FORB = 416
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
#define DIMPORT 283
#define DALIAS 284
#define DSWITCH 285
#define DSELECT 286
#define DCASE 287
#define DENDSWITCH 288
#define DLINE 289
#define DSTRING 290
#define DISTRING 291
#define DCSTRING 292
#define DHAS 293
#define DHASNT 294
#define DINHERIT 295
#define DINSTANCE 296
#define SYMBOL 297
#define EXPORT 298
#define LABEL 299
#define INTEGER 300
#define REG_A 301
#define REG_B 302
#define REG_S1 303
#define REG_S2 304
#define REG_L1 305
#define REG_L2 306
#define NUMERIC 307
#define STRING 308
#define STRING_ID 309
#define TRUE_TOKEN 310
#define FALSE_TOKEN 311
#define I_LD 312
#define I_LNIL 313
#define NIL 314
#define I_ADD 315
#define I_SUB 316
#define I_MUL 317
#define I_DIV 318
#define I_MOD 319
#define I_POW 320
#define I_ADDS 321
#define I_SUBS 322
#define I_MULS 323
#define I_DIVS 324
#define I_POWS 325
#define I_INC 326
#define I_DEC 327
#define I_INCP 328
#define I_DECP 329
#define I_NEG 330
#define I_NOT 331
#define I_RET 332
#define I_RETV 333
#define I_RETA 334
#define I_FORK 335
#define I_PUSH 336
#define I_PSHR 337
#define I_PSHN 338
#define I_POP 339
#define I_LDV 340
#define I_LDVT 341
#define I_STV 342
#define I_STVR 343
#define I_STVS 344
#define I_LDP 345
#define I_LDPT 346
#define I_STP 347
#define I_STPR 348
#define I_STPS 349
#define I_TRAV 350
#define I_TRAN 351
#define I_TRAL 352
#define I_IPOP 353
#define I_XPOP 354
#define I_GENA 355
#define I_GEND 356
#define I_GENR 357
#define I_GEOR 358
#define I_JMP 359
#define I_IFT 360
#define I_IFF 361
#define I_BOOL 362
#define I_EQ 363
#define I_NEQ 364
#define I_GT 365
#define I_GE 366
#define I_LT 367
#define I_LE 368
#define I_UNPK 369
#define I_UNPS 370
#define I_CALL 371
#define I_INST 372
#define I_SWCH 373
#define I_SELE 374
#define I_NOP 375
#define I_TRY 376
#define I_JTRY 377
#define I_PTRY 378
#define I_RIS 379
#define I_LDRF 380
#define I_ONCE 381
#define I_BAND 382
#define I_BOR 383
#define I_BXOR 384
#define I_BNOT 385
#define I_MODS 386
#define I_AND 387
#define I_OR 388
#define I_ANDS 389
#define I_ORS 390
#define I_XORS 391
#define I_NOTS 392
#define I_HAS 393
#define I_HASN 394
#define I_GIVE 395
#define I_GIVN 396
#define I_IN 397
#define I_NOIN 398
#define I_PROV 399
#define I_END 400
#define I_PEEK 401
#define I_PSIN 402
#define I_PASS 403
#define I_SHR 404
#define I_SHL 405
#define I_SHRS 406
#define I_SHLS 407
#define I_LDVR 408
#define I_LDPR 409
#define I_LSB 410
#define I_INDI 411
#define I_STEX 412
#define I_TRAC 413
#define I_WRT 414
#define I_STO 415
#define I_FORB 416




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



