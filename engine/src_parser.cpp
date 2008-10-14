/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse flc_src_parse
#define yylex   flc_src_lex
#define yyerror flc_src_error
#define yylval  flc_src_lval
#define yychar  flc_src_char
#define yydebug flc_src_debug
#define yynerrs flc_src_nerrs


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
     FORMIDDLE = 276,
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
     IMPORT = 300,
     DIRECTIVE = 301,
     COLON = 302,
     FUNCDECL = 303,
     STATIC = 304,
     INNERFUNC = 305,
     FORDOT = 306,
     LISTPAR = 307,
     LOOP = 308,
     ENUM = 309,
     TRUE_TOKEN = 310,
     FALSE_TOKEN = 311,
     OUTER_STRING = 312,
     CLOSEPAR = 313,
     OPENPAR = 314,
     CLOSESQUARE = 315,
     OPENSQUARE = 316,
     DOT = 317,
     ARROW = 318,
     VBAR = 319,
     ASSIGN_POW = 320,
     ASSIGN_SHL = 321,
     ASSIGN_SHR = 322,
     ASSIGN_BXOR = 323,
     ASSIGN_BOR = 324,
     ASSIGN_BAND = 325,
     ASSIGN_MOD = 326,
     ASSIGN_DIV = 327,
     ASSIGN_MUL = 328,
     ASSIGN_SUB = 329,
     ASSIGN_ADD = 330,
     OP_EQ = 331,
     OP_AS = 332,
     OP_TO = 333,
     COMMA = 334,
     QUESTION = 335,
     OR = 336,
     AND = 337,
     NOT = 338,
     LE = 339,
     GE = 340,
     LT = 341,
     GT = 342,
     NEQ = 343,
     EEQ = 344,
     PROVIDES = 345,
     OP_NOTIN = 346,
     OP_IN = 347,
     HASNT = 348,
     HAS = 349,
     DIESIS = 350,
     ATSIGN = 351,
     CAP_CAP = 352,
     VBAR_VBAR = 353,
     AMPER_AMPER = 354,
     MINUS = 355,
     PLUS = 356,
     PERCENT = 357,
     SLASH = 358,
     STAR = 359,
     POW = 360,
     SHR = 361,
     SHL = 362,
     TILDE = 363,
     NEG = 364,
     AMPER = 365,
     BANG = 366,
     DECREMENT = 367,
     INCREMENT = 368,
     DOLLAR = 369
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
#define FORMIDDLE 276
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
#define IMPORT 300
#define DIRECTIVE 301
#define COLON 302
#define FUNCDECL 303
#define STATIC 304
#define INNERFUNC 305
#define FORDOT 306
#define LISTPAR 307
#define LOOP 308
#define ENUM 309
#define TRUE_TOKEN 310
#define FALSE_TOKEN 311
#define OUTER_STRING 312
#define CLOSEPAR 313
#define OPENPAR 314
#define CLOSESQUARE 315
#define OPENSQUARE 316
#define DOT 317
#define ARROW 318
#define VBAR 319
#define ASSIGN_POW 320
#define ASSIGN_SHL 321
#define ASSIGN_SHR 322
#define ASSIGN_BXOR 323
#define ASSIGN_BOR 324
#define ASSIGN_BAND 325
#define ASSIGN_MOD 326
#define ASSIGN_DIV 327
#define ASSIGN_MUL 328
#define ASSIGN_SUB 329
#define ASSIGN_ADD 330
#define OP_EQ 331
#define OP_AS 332
#define OP_TO 333
#define COMMA 334
#define QUESTION 335
#define OR 336
#define AND 337
#define NOT 338
#define LE 339
#define GE 340
#define LT 341
#define GT 342
#define NEQ 343
#define EEQ 344
#define PROVIDES 345
#define OP_NOTIN 346
#define OP_IN 347
#define HASNT 348
#define HAS 349
#define DIESIS 350
#define ATSIGN 351
#define CAP_CAP 352
#define VBAR_VBAR 353
#define AMPER_AMPER 354
#define MINUS 355
#define PLUS 356
#define PERCENT 357
#define SLASH 358
#define STAR 359
#define POW 360
#define SHR 361
#define SHL 362
#define TILDE 363
#define NEG 364
#define AMPER 365
#define BANG 366
#define DECREMENT 367
#define INCREMENT 368
#define DOLLAR 369




/* Copy the first part of user declarations.  */
#line 17 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


#include <math.h>
#include <stdio.h>
#include <iostream>

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <stdlib.h>

#include <falcon/fassert.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::Compiler *>(yyparam) )
#define  CTX_LINE  ( COMPILER->lexer()->ctxOpenLine() )
#define  LINE      ( COMPILER->lexer()->previousLine() )
#define  CURRENT_LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM yyparam
#define YYLEX_PARAM yyparam

int flc_src_parse( void *param );
void flc_src_error (const char *s);

inline int flc_src_lex (void *lvalp, void *yyparam)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
   Falcon::List *fal_genericList;
}
/* Line 187 of yacc.c.  */
#line 389 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 402 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   6176

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  115
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  164
/* YYNRULES -- Number of rules.  */
#define YYNRULES  453
/* YYNRULES -- Number of states.  */
#define YYNSTATES  830

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   369

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      42,    45,    49,    53,    57,    59,    61,    65,    69,    73,
      76,    79,    84,    91,    93,    95,    97,    99,   101,   103,
     105,   107,   109,   111,   113,   115,   117,   119,   121,   123,
     125,   127,   131,   137,   141,   145,   146,   152,   155,   159,
     161,   165,   169,   172,   176,   177,   184,   187,   191,   195,
     199,   203,   204,   206,   207,   211,   214,   218,   219,   224,
     228,   232,   233,   236,   239,   243,   246,   250,   254,   255,
     265,   266,   274,   280,   284,   285,   288,   290,   292,   294,
     296,   300,   304,   308,   311,   315,   318,   322,   326,   328,
     329,   336,   340,   344,   345,   352,   356,   360,   361,   368,
     372,   376,   377,   384,   388,   392,   393,   396,   400,   402,
     403,   409,   410,   416,   417,   423,   424,   430,   431,   432,
     436,   437,   439,   442,   445,   448,   450,   454,   456,   458,
     460,   464,   466,   467,   474,   478,   482,   483,   486,   490,
     492,   493,   499,   500,   506,   507,   513,   514,   520,   522,
     526,   527,   529,   531,   537,   542,   546,   550,   551,   558,
     561,   565,   566,   568,   570,   573,   576,   579,   584,   588,
     594,   598,   600,   604,   606,   608,   612,   616,   622,   625,
     631,   632,   640,   644,   650,   651,   658,   661,   662,   664,
     668,   670,   671,   672,   678,   679,   683,   686,   690,   693,
     697,   701,   705,   709,   715,   721,   725,   731,   737,   741,
     744,   748,   752,   754,   758,   762,   768,   774,   782,   790,
     798,   806,   811,   816,   821,   826,   833,   840,   844,   846,
     850,   854,   858,   860,   864,   868,   872,   876,   881,   885,
     888,   892,   895,   899,   900,   902,   906,   909,   913,   916,
     917,   926,   930,   933,   934,   938,   939,   945,   946,   949,
     951,   955,   958,   959,   963,   965,   969,   971,   973,   975,
     976,   979,   981,   983,   985,   987,   988,   996,  1002,  1007,
    1008,  1012,  1016,  1018,  1021,  1025,  1030,  1031,  1039,  1040,
    1043,  1045,  1050,  1053,  1055,  1057,  1058,  1067,  1070,  1073,
    1074,  1077,  1079,  1081,  1083,  1085,  1086,  1091,  1093,  1097,
    1101,  1103,  1106,  1110,  1114,  1116,  1118,  1120,  1122,  1124,
    1126,  1128,  1130,  1132,  1134,  1136,  1138,  1141,  1144,  1147,
    1151,  1155,  1159,  1163,  1167,  1171,  1175,  1179,  1183,  1187,
    1191,  1195,  1198,  1202,  1205,  1208,  1211,  1214,  1218,  1222,
    1226,  1230,  1234,  1238,  1242,  1245,  1249,  1253,  1257,  1261,
    1265,  1268,  1271,  1274,  1277,  1279,  1281,  1283,  1285,  1287,
    1289,  1292,  1294,  1299,  1305,  1309,  1311,  1313,  1317,  1323,
    1327,  1331,  1335,  1339,  1343,  1347,  1351,  1355,  1359,  1363,
    1367,  1371,  1375,  1380,  1385,  1391,  1399,  1404,  1408,  1409,
    1416,  1417,  1424,  1429,  1433,  1436,  1437,  1444,  1445,  1451,
    1453,  1456,  1462,  1468,  1473,  1477,  1480,  1484,  1488,  1491,
    1495,  1499,  1503,  1507,  1512,  1514,  1518,  1520,  1524,  1525,
    1527,  1529,  1533,  1537
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     116,     0,    -1,   117,    -1,    -1,   117,   118,    -1,   119,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   121,    -1,
     214,    -1,   194,    -1,   222,    -1,   245,    -1,   240,    -1,
     122,    -1,   209,    -1,   210,    -1,   212,    -1,   217,    -1,
       4,    -1,   100,     4,    -1,    39,     6,     3,    -1,    39,
       7,     3,    -1,    39,     1,     3,    -1,   123,    -1,     3,
      -1,    48,     1,     3,    -1,    34,     1,     3,    -1,    32,
       1,     3,    -1,     1,     3,    -1,   258,     3,    -1,   274,
      76,   258,     3,    -1,   274,    76,   258,    79,   274,     3,
      -1,   125,    -1,   126,    -1,   143,    -1,   157,    -1,   172,
      -1,   130,    -1,   141,    -1,   142,    -1,   183,    -1,   184,
      -1,   193,    -1,   254,    -1,   250,    -1,   207,    -1,   208,
      -1,   148,    -1,   149,    -1,   150,    -1,   256,    76,   258,
      -1,   124,    79,   256,    76,   258,    -1,    10,   124,     3,
      -1,    10,     1,     3,    -1,    -1,   128,   127,   140,     9,
       3,    -1,   129,   122,    -1,    11,   258,     3,    -1,    53,
      -1,    11,     1,     3,    -1,    11,   258,    47,    -1,    53,
      47,    -1,    11,     1,    47,    -1,    -1,   132,   131,   140,
     134,     9,     3,    -1,   133,   122,    -1,    15,   258,     3,
      -1,    15,     1,     3,    -1,    15,   258,    47,    -1,    15,
       1,    47,    -1,    -1,   137,    -1,    -1,   136,   135,   140,
      -1,    16,     3,    -1,    16,     1,     3,    -1,    -1,   139,
     138,   140,   134,    -1,    17,   258,     3,    -1,    17,     1,
       3,    -1,    -1,   140,   122,    -1,    12,     3,    -1,    12,
       1,     3,    -1,    13,     3,    -1,    13,    14,     3,    -1,
      13,     1,     3,    -1,    -1,    18,   277,    92,   258,     3,
     144,   146,     9,     3,    -1,    -1,    18,   277,    92,   258,
      47,   145,   122,    -1,    18,   277,    92,     1,     3,    -1,
      18,     1,     3,    -1,    -1,   147,   146,    -1,   122,    -1,
     151,    -1,   153,    -1,   155,    -1,    51,   258,     3,    -1,
      51,     1,     3,    -1,   106,   274,     3,    -1,   106,     3,
      -1,    87,   274,     3,    -1,    87,     3,    -1,   106,     1,
       3,    -1,    87,     1,     3,    -1,    57,    -1,    -1,    19,
       3,   152,   140,     9,     3,    -1,    19,    47,   122,    -1,
      19,     1,     3,    -1,    -1,    20,     3,   154,   140,     9,
       3,    -1,    20,    47,   122,    -1,    20,     1,     3,    -1,
      -1,    21,     3,   156,   140,     9,     3,    -1,    21,    47,
     122,    -1,    21,     1,     3,    -1,    -1,   159,   158,   160,
     166,     9,     3,    -1,    22,   258,     3,    -1,    22,     1,
       3,    -1,    -1,   160,   161,    -1,   160,     1,     3,    -1,
       3,    -1,    -1,    23,   170,     3,   162,   140,    -1,    -1,
      23,   170,    47,   163,   122,    -1,    -1,    23,     1,     3,
     164,   140,    -1,    -1,    23,     1,    47,   165,   122,    -1,
      -1,    -1,   168,   167,   169,    -1,    -1,    24,    -1,    24,
       1,    -1,     3,   140,    -1,    47,   122,    -1,   171,    -1,
     170,    79,   171,    -1,     8,    -1,   120,    -1,     7,    -1,
     120,    78,   120,    -1,     6,    -1,    -1,   174,   173,   175,
     166,     9,     3,    -1,    25,   258,     3,    -1,    25,     1,
       3,    -1,    -1,   175,   176,    -1,   175,     1,     3,    -1,
       3,    -1,    -1,    23,   181,     3,   177,   140,    -1,    -1,
      23,   181,    47,   178,   122,    -1,    -1,    23,     1,     3,
     179,   140,    -1,    -1,    23,     1,    47,   180,   122,    -1,
     182,    -1,   181,    79,   182,    -1,    -1,     4,    -1,     6,
      -1,    28,   274,    78,   274,     3,    -1,    28,   274,     1,
       3,    -1,    28,     1,     3,    -1,    29,    47,   122,    -1,
      -1,   186,   185,   140,   187,     9,     3,    -1,    29,     3,
      -1,    29,     1,     3,    -1,    -1,   188,    -1,   189,    -1,
     188,   189,    -1,   190,   140,    -1,    30,     3,    -1,    30,
      92,   256,     3,    -1,    30,   191,     3,    -1,    30,   191,
      92,   256,     3,    -1,    30,     1,     3,    -1,   192,    -1,
     191,    79,   192,    -1,     4,    -1,     6,    -1,    31,   258,
       3,    -1,    31,     1,     3,    -1,   195,   202,   140,     9,
       3,    -1,   197,   122,    -1,   199,    59,   200,    58,     3,
      -1,    -1,   199,    59,   200,     1,   196,    58,     3,    -1,
     199,     1,     3,    -1,   199,    59,   200,    58,    47,    -1,
      -1,   199,    59,     1,   198,    58,    47,    -1,    48,     6,
      -1,    -1,   201,    -1,   200,    79,   201,    -1,     6,    -1,
      -1,    -1,   205,   203,   140,     9,     3,    -1,    -1,   206,
     204,   122,    -1,    49,     3,    -1,    49,     1,     3,    -1,
      49,    47,    -1,    49,     1,    47,    -1,    40,   260,     3,
      -1,    40,     1,     3,    -1,    43,   258,     3,    -1,    43,
     258,    92,   258,     3,    -1,    43,   258,    92,     1,     3,
      -1,    43,     1,     3,    -1,    41,     6,    76,   255,     3,
      -1,    41,     6,    76,     1,     3,    -1,    41,     1,     3,
      -1,    44,     3,    -1,    44,   211,     3,    -1,    44,     1,
       3,    -1,     6,    -1,   211,    79,     6,    -1,    45,   213,
       3,    -1,    45,   213,    33,     6,     3,    -1,    45,   213,
      33,     7,     3,    -1,    45,   213,    33,     6,    77,     6,
       3,    -1,    45,   213,    33,     7,    77,     6,     3,    -1,
      45,   213,    33,     6,    78,     6,     3,    -1,    45,   213,
      33,     7,    78,     6,     3,    -1,    45,     6,     1,     3,
      -1,    45,   213,     1,     3,    -1,    45,    33,     6,     3,
      -1,    45,    33,     7,     3,    -1,    45,    33,     6,    76,
       6,     3,    -1,    45,    33,     7,    76,     6,     3,    -1,
      45,     1,     3,    -1,     6,    -1,   213,    79,     6,    -1,
      46,   215,     3,    -1,    46,     1,     3,    -1,   216,    -1,
     215,    79,   216,    -1,     6,    76,     6,    -1,     6,    76,
       7,    -1,     6,    76,   120,    -1,   218,   221,     9,     3,
      -1,   219,   220,     3,    -1,    42,     3,    -1,    42,     1,
       3,    -1,    42,    47,    -1,    42,     1,    47,    -1,    -1,
       6,    -1,   220,    79,     6,    -1,   220,     3,    -1,   221,
     220,     3,    -1,     1,     3,    -1,    -1,    32,     6,   223,
     224,   233,   238,     9,     3,    -1,   225,   227,     3,    -1,
       1,     3,    -1,    -1,    59,   200,    58,    -1,    -1,    59,
     200,     1,   226,    58,    -1,    -1,    33,   228,    -1,   229,
      -1,   228,    79,   229,    -1,     6,   230,    -1,    -1,    59,
     231,    58,    -1,   232,    -1,   231,    79,   232,    -1,   255,
      -1,     6,    -1,    27,    -1,    -1,   233,   234,    -1,     3,
      -1,   194,    -1,   237,    -1,   235,    -1,    -1,    38,     3,
     236,   202,   140,     9,     3,    -1,    49,     6,    76,   258,
       3,    -1,     6,    76,   258,     3,    -1,    -1,    94,   239,
       3,    -1,    94,     1,     3,    -1,     6,    -1,    83,     6,
      -1,   239,    79,     6,    -1,   239,    79,    83,     6,    -1,
      -1,    54,     6,   241,     3,   242,     9,     3,    -1,    -1,
     242,   243,    -1,     3,    -1,     6,    76,   255,   244,    -1,
       6,   244,    -1,     3,    -1,    79,    -1,    -1,    34,     6,
     246,   247,   248,   238,     9,     3,    -1,   227,     3,    -1,
       1,     3,    -1,    -1,   248,   249,    -1,     3,    -1,   194,
      -1,   237,    -1,   235,    -1,    -1,    36,   251,   252,     3,
      -1,   253,    -1,   252,    79,   253,    -1,   252,    79,     1,
      -1,     6,    -1,    35,     3,    -1,    35,   258,     3,    -1,
      35,     1,     3,    -1,     8,    -1,    55,    -1,    56,    -1,
       4,    -1,     5,    -1,     7,    -1,     6,    -1,   256,    -1,
      27,    -1,    26,    -1,   255,    -1,   257,    -1,   110,     6,
      -1,   110,    27,    -1,   100,   258,    -1,     6,    64,   258,
      -1,   258,   101,   258,    -1,   258,   100,   258,    -1,   258,
     104,   258,    -1,   258,   103,   258,    -1,   258,   102,   258,
      -1,   258,   105,   258,    -1,   258,    99,   258,    -1,   258,
      98,   258,    -1,   258,    97,   258,    -1,   258,   107,   258,
      -1,   258,   106,   258,    -1,   108,   258,    -1,   258,    88,
     258,    -1,   258,   113,    -1,   113,   258,    -1,   258,   112,
      -1,   112,   258,    -1,   258,    89,   258,    -1,   258,    87,
     258,    -1,   258,    86,   258,    -1,   258,    85,   258,    -1,
     258,    84,   258,    -1,   258,    82,   258,    -1,   258,    81,
     258,    -1,    83,   258,    -1,   258,    94,   258,    -1,   258,
      93,   258,    -1,   258,    92,   258,    -1,   258,    91,   258,
      -1,   258,    90,     6,    -1,   114,   256,    -1,   114,     4,
      -1,    96,   258,    -1,    95,   258,    -1,   267,    -1,   262,
      -1,   265,    -1,   260,    -1,   270,    -1,   272,    -1,   258,
     259,    -1,   271,    -1,   258,    61,   258,    60,    -1,   258,
      61,   104,   258,    60,    -1,   258,    62,     6,    -1,   273,
      -1,   259,    -1,   258,    76,   258,    -1,   258,    76,   258,
      79,   274,    -1,   258,    75,   258,    -1,   258,    74,   258,
      -1,   258,    73,   258,    -1,   258,    72,   258,    -1,   258,
      71,   258,    -1,   258,    65,   258,    -1,   258,    70,   258,
      -1,   258,    69,   258,    -1,   258,    68,   258,    -1,   258,
      66,   258,    -1,   258,    67,   258,    -1,    59,   258,    58,
      -1,    61,    47,    60,    -1,    61,   258,    47,    60,    -1,
      61,    47,   258,    60,    -1,    61,   258,    47,   258,    60,
      -1,    61,   258,    47,   258,    47,   258,    60,    -1,   258,
      59,   274,    58,    -1,   258,    59,    58,    -1,    -1,   258,
      59,   274,     1,   261,    58,    -1,    -1,    48,   263,   264,
     202,   140,     9,    -1,    59,   200,    58,     3,    -1,    59,
     200,     1,    -1,     1,     3,    -1,    -1,    50,   266,   264,
     202,   140,     9,    -1,    -1,    37,   268,   269,    63,   258,
      -1,   200,    -1,     1,     3,    -1,   258,    80,   258,    47,
     258,    -1,   258,    80,   258,    47,     1,    -1,   258,    80,
     258,     1,    -1,   258,    80,     1,    -1,    61,    60,    -1,
      61,   274,    60,    -1,    61,   274,     1,    -1,    52,    60,
      -1,    52,   275,    60,    -1,    52,   275,     1,    -1,    61,
      63,    60,    -1,    61,   278,    60,    -1,    61,   278,     1,
      60,    -1,   258,    -1,   274,    79,   258,    -1,   258,    -1,
     275,   276,   258,    -1,    -1,    79,    -1,   256,    -1,   277,
      79,   256,    -1,   258,    63,   258,    -1,   278,    79,   258,
      63,   258,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   200,   200,   203,   205,   209,   210,   211,   215,   216,
     217,   222,   227,   232,   237,   242,   243,   244,   245,   249,
     250,   254,   260,   266,   273,   274,   275,   276,   277,   278,
     283,   285,   291,   305,   306,   307,   308,   309,   310,   311,
     312,   313,   314,   315,   316,   317,   318,   319,   320,   321,
     322,   326,   332,   340,   342,   347,   347,   361,   369,   370,
     371,   375,   376,   377,   381,   381,   396,   406,   407,   411,
     412,   416,   418,   419,   419,   428,   429,   434,   434,   446,
     447,   450,   452,   458,   467,   475,   485,   494,   504,   503,
     528,   527,   553,   558,   565,   567,   571,   578,   579,   580,
     584,   597,   605,   609,   615,   621,   628,   633,   642,   652,
     652,   666,   675,   679,   679,   692,   701,   705,   705,   721,
     730,   734,   734,   751,   752,   759,   761,   762,   766,   768,
     767,   778,   778,   790,   790,   802,   802,   818,   821,   820,
     833,   834,   835,   838,   839,   845,   846,   850,   859,   871,
     882,   893,   914,   914,   931,   932,   939,   941,   942,   946,
     948,   947,   958,   958,   971,   971,   983,   983,  1001,  1002,
    1005,  1006,  1018,  1039,  1043,  1048,  1056,  1063,  1062,  1081,
    1082,  1085,  1087,  1091,  1092,  1096,  1101,  1119,  1139,  1149,
    1160,  1168,  1169,  1173,  1185,  1208,  1209,  1216,  1226,  1235,
    1236,  1236,  1240,  1244,  1245,  1245,  1252,  1306,  1308,  1309,
    1313,  1328,  1331,  1330,  1342,  1341,  1356,  1357,  1361,  1362,
    1371,  1375,  1383,  1390,  1405,  1411,  1423,  1433,  1438,  1450,
    1459,  1466,  1474,  1479,  1487,  1492,  1497,  1502,  1522,  1541,
    1546,  1551,  1556,  1570,  1575,  1580,  1585,  1590,  1598,  1604,
    1616,  1621,  1629,  1630,  1634,  1638,  1642,  1654,  1661,  1671,
    1672,  1675,  1676,  1679,  1681,  1685,  1692,  1693,  1694,  1706,
    1705,  1764,  1767,  1773,  1775,  1776,  1776,  1782,  1784,  1788,
    1789,  1793,  1827,  1829,  1838,  1839,  1843,  1844,  1853,  1856,
    1858,  1862,  1863,  1866,  1884,  1888,  1888,  1920,  1942,  1969,
    1971,  1972,  1979,  1987,  1993,  1999,  2013,  2012,  2056,  2058,
    2062,  2063,  2068,  2075,  2075,  2084,  2083,  2149,  2150,  2156,
    2158,  2162,  2163,  2166,  2185,  2194,  2193,  2211,  2212,  2213,
    2220,  2236,  2237,  2238,  2248,  2249,  2250,  2251,  2252,  2253,
    2257,  2275,  2276,  2277,  2288,  2289,  2290,  2291,  2292,  2293,
    2294,  2295,  2296,  2297,  2298,  2299,  2300,  2301,  2302,  2303,
    2304,  2305,  2306,  2307,  2308,  2309,  2310,  2311,  2312,  2313,
    2314,  2315,  2316,  2317,  2318,  2319,  2320,  2321,  2322,  2323,
    2324,  2325,  2326,  2327,  2328,  2329,  2330,  2331,  2332,  2333,
    2335,  2340,  2344,  2349,  2355,  2364,  2365,  2367,  2372,  2379,
    2380,  2381,  2382,  2383,  2384,  2385,  2386,  2387,  2388,  2389,
    2390,  2395,  2398,  2401,  2404,  2407,  2413,  2419,  2424,  2424,
    2434,  2433,  2474,  2475,  2479,  2487,  2486,  2532,  2531,  2573,
    2574,  2583,  2588,  2595,  2602,  2612,  2613,  2617,  2625,  2626,
    2630,  2639,  2640,  2641,  2649,  2650,  2654,  2655,  2658,  2659,
    2662,  2668,  2675,  2676
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "END", "DEF", "WHILE", "BREAK", "CONTINUE", "DROPPING",
  "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST", "FORMIDDLE",
  "SWITCH", "CASE", "DEFAULT", "SELECT", "SENDER", "SELF", "GIVE", "TRY",
  "CATCH", "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL",
  "LAMBDA", "INIT", "LOAD", "LAUNCH", "CONST_KW", "ATTRIBUTES", "PASS",
  "EXPORT", "IMPORT", "DIRECTIVE", "COLON", "FUNCDECL", "STATIC",
  "INNERFUNC", "FORDOT", "LISTPAR", "LOOP", "ENUM", "TRUE_TOKEN",
  "FALSE_TOKEN", "OUTER_STRING", "CLOSEPAR", "OPENPAR", "CLOSESQUARE",
  "OPENSQUARE", "DOT", "ARROW", "VBAR", "ASSIGN_POW", "ASSIGN_SHL",
  "ASSIGN_SHR", "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD",
  "ASSIGN_DIV", "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_AS",
  "OP_TO", "COMMA", "QUESTION", "OR", "AND", "NOT", "LE", "GE", "LT", "GT",
  "NEQ", "EEQ", "PROVIDES", "OP_NOTIN", "OP_IN", "HASNT", "HAS", "DIESIS",
  "ATSIGN", "CAP_CAP", "VBAR_VBAR", "AMPER_AMPER", "MINUS", "PLUS",
  "PERCENT", "SLASH", "STAR", "POW", "SHR", "SHL", "TILDE", "NEG", "AMPER",
  "BANG", "DECREMENT", "INCREMENT", "DOLLAR", "$accept", "input", "body",
  "line", "toplevel_statement", "INTNUM_WITH_MINUS", "load_statement",
  "statement", "base_statement", "assignment_def_list", "def_statement",
  "while_statement", "@1", "while_decl", "while_short_decl",
  "if_statement", "@2", "if_decl", "if_short_decl", "elif_or_else", "@3",
  "else_decl", "elif_statement", "@4", "elif_decl", "statement_list",
  "break_statement", "continue_statement", "forin_statement", "@5", "@6",
  "forin_statement_list", "forin_statement_elem", "fordot_statement",
  "self_print_statement", "outer_print_statement", "first_loop_block",
  "@7", "last_loop_block", "@8", "middle_loop_block", "@9",
  "switch_statement", "@10", "switch_decl", "case_list", "case_statement",
  "@11", "@12", "@13", "@14", "default_statement", "@15", "default_decl",
  "default_body", "case_expression_list", "case_element",
  "select_statement", "@16", "select_decl", "selcase_list",
  "selcase_statement", "@17", "@18", "@19", "@20",
  "selcase_expression_list", "selcase_element", "give_statement",
  "try_statement", "@21", "try_decl", "catch_statements", "catch_list",
  "catch_statement", "catch_decl", "catchcase_element_list",
  "catchcase_element", "raise_statement", "func_statement", "func_decl",
  "@22", "func_decl_short", "@23", "func_begin", "param_list",
  "param_symbol", "static_block", "@24", "@25", "static_decl",
  "static_short_decl", "launch_statement", "pass_statement",
  "const_statement", "export_statement", "export_symbol_list",
  "import_statement", "import_symbol_list", "directive_statement",
  "directive_pair_list", "directive_pair", "attributes_statement",
  "attributes_decl", "attributes_short_decl", "attribute_list",
  "attribute_vert_list", "class_decl", "@26", "class_def_inner",
  "class_param_list", "@27", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "inherit_param_list",
  "inherit_param_token", "class_statement_list", "class_statement",
  "init_decl", "@28", "property_decl", "has_list", "has_clause_list",
  "enum_statement", "@29", "enum_statement_list", "enum_item_decl",
  "enum_item_terminator", "object_decl", "@30", "object_decl_inner",
  "object_statement_list", "object_statement", "global_statement", "@31",
  "global_symbol_list", "globalized_symbol", "return_statement",
  "const_atom", "atomic_symbol", "var_atom", "expression", "range_decl",
  "func_call", "@32", "nameless_func", "@33", "nameless_func_decl_inner",
  "nameless_closure", "@34", "lambda_expr", "@35", "lambda_expr_inner",
  "iif_expr", "array_decl", "dotarray_decl", "dict_decl",
  "expression_list", "listpar_expression_list", "listpar_comma",
  "symbol_list", "expression_pair_list", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   115,   116,   117,   117,   118,   118,   118,   119,   119,
     119,   119,   119,   119,   119,   119,   119,   119,   119,   120,
     120,   121,   121,   121,   122,   122,   122,   122,   122,   122,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     123,   124,   124,   125,   125,   127,   126,   126,   128,   128,
     128,   129,   129,   129,   131,   130,   130,   132,   132,   133,
     133,   134,   134,   135,   134,   136,   136,   138,   137,   139,
     139,   140,   140,   141,   141,   142,   142,   142,   144,   143,
     145,   143,   143,   143,   146,   146,   147,   147,   147,   147,
     148,   148,   149,   149,   149,   149,   149,   149,   150,   152,
     151,   151,   151,   154,   153,   153,   153,   156,   155,   155,
     155,   158,   157,   159,   159,   160,   160,   160,   161,   162,
     161,   163,   161,   164,   161,   165,   161,   166,   167,   166,
     168,   168,   168,   169,   169,   170,   170,   171,   171,   171,
     171,   171,   173,   172,   174,   174,   175,   175,   175,   176,
     177,   176,   178,   176,   179,   176,   180,   176,   181,   181,
     182,   182,   182,   183,   183,   183,   184,   185,   184,   186,
     186,   187,   187,   188,   188,   189,   190,   190,   190,   190,
     190,   191,   191,   192,   192,   193,   193,   194,   194,   195,
     196,   195,   195,   197,   198,   197,   199,   200,   200,   200,
     201,   202,   203,   202,   204,   202,   205,   205,   206,   206,
     207,   207,   208,   208,   208,   208,   209,   209,   209,   210,
     210,   210,   211,   211,   212,   212,   212,   212,   212,   212,
     212,   212,   212,   212,   212,   212,   212,   212,   213,   213,
     214,   214,   215,   215,   216,   216,   216,   217,   217,   218,
     218,   219,   219,   220,   220,   220,   221,   221,   221,   223,
     222,   224,   224,   225,   225,   226,   225,   227,   227,   228,
     228,   229,   230,   230,   231,   231,   232,   232,   232,   233,
     233,   234,   234,   234,   234,   236,   235,   237,   237,   238,
     238,   238,   239,   239,   239,   239,   241,   240,   242,   242,
     243,   243,   243,   244,   244,   246,   245,   247,   247,   248,
     248,   249,   249,   249,   249,   251,   250,   252,   252,   252,
     253,   254,   254,   254,   255,   255,   255,   255,   255,   255,
     256,   257,   257,   257,   258,   258,   258,   258,   258,   258,
     258,   258,   258,   258,   258,   258,   258,   258,   258,   258,
     258,   258,   258,   258,   258,   258,   258,   258,   258,   258,
     258,   258,   258,   258,   258,   258,   258,   258,   258,   258,
     258,   258,   258,   258,   258,   258,   258,   258,   258,   258,
     258,   258,   258,   258,   258,   258,   258,   258,   258,   258,
     258,   258,   258,   258,   258,   258,   258,   258,   258,   258,
     258,   259,   259,   259,   259,   259,   260,   260,   261,   260,
     263,   262,   264,   264,   264,   266,   265,   268,   267,   269,
     269,   270,   270,   270,   270,   271,   271,   271,   272,   272,
     272,   273,   273,   273,   274,   274,   275,   275,   276,   276,
     277,   277,   278,   278
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     3,     3,     3,     1,     1,     3,     3,     3,     2,
       2,     4,     6,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     5,     3,     3,     0,     5,     2,     3,     1,
       3,     3,     2,     3,     0,     6,     2,     3,     3,     3,
       3,     0,     1,     0,     3,     2,     3,     0,     4,     3,
       3,     0,     2,     2,     3,     2,     3,     3,     0,     9,
       0,     7,     5,     3,     0,     2,     1,     1,     1,     1,
       3,     3,     3,     2,     3,     2,     3,     3,     1,     0,
       6,     3,     3,     0,     6,     3,     3,     0,     6,     3,
       3,     0,     6,     3,     3,     0,     2,     3,     1,     0,
       5,     0,     5,     0,     5,     0,     5,     0,     0,     3,
       0,     1,     2,     2,     2,     1,     3,     1,     1,     1,
       3,     1,     0,     6,     3,     3,     0,     2,     3,     1,
       0,     5,     0,     5,     0,     5,     0,     5,     1,     3,
       0,     1,     1,     5,     4,     3,     3,     0,     6,     2,
       3,     0,     1,     1,     2,     2,     2,     4,     3,     5,
       3,     1,     3,     1,     1,     3,     3,     5,     2,     5,
       0,     7,     3,     5,     0,     6,     2,     0,     1,     3,
       1,     0,     0,     5,     0,     3,     2,     3,     2,     3,
       3,     3,     3,     5,     5,     3,     5,     5,     3,     2,
       3,     3,     1,     3,     3,     5,     5,     7,     7,     7,
       7,     4,     4,     4,     4,     6,     6,     3,     1,     3,
       3,     3,     1,     3,     3,     3,     3,     4,     3,     2,
       3,     2,     3,     0,     1,     3,     2,     3,     2,     0,
       8,     3,     2,     0,     3,     0,     5,     0,     2,     1,
       3,     2,     0,     3,     1,     3,     1,     1,     1,     0,
       2,     1,     1,     1,     1,     0,     7,     5,     4,     0,
       3,     3,     1,     2,     3,     4,     0,     7,     0,     2,
       1,     4,     2,     1,     1,     0,     8,     2,     2,     0,
       2,     1,     1,     1,     1,     0,     4,     1,     3,     3,
       1,     2,     3,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     2,     2,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     2,     3,     2,     2,     2,     2,     3,     3,     3,
       3,     3,     3,     3,     2,     3,     3,     3,     3,     3,
       2,     2,     2,     2,     1,     1,     1,     1,     1,     1,
       2,     1,     4,     5,     3,     1,     1,     3,     5,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     4,     4,     5,     7,     4,     3,     0,     6,
       0,     6,     4,     3,     2,     0,     6,     0,     5,     1,
       2,     5,     5,     4,     3,     2,     3,     3,     2,     3,
       3,     3,     3,     4,     1,     3,     1,     3,     0,     1,
       1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    25,   337,   338,   340,   339,
     334,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   343,   342,     0,     0,     0,     0,     0,     0,   325,
     427,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     425,     0,     0,    59,     0,   335,   336,   108,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     4,     5,     8,    14,    24,    33,    34,    55,     0,
      38,    64,     0,    39,    40,    35,    48,    49,    50,    36,
     121,    37,   152,    41,    42,   177,    43,    10,   211,     0,
       0,    46,    47,    15,    16,    17,     9,    18,     0,   263,
      11,    13,    12,    45,    44,   344,   341,   345,   444,   396,
     387,   385,   386,   384,   388,   391,   389,   395,     0,    29,
       0,     6,     0,   340,     0,     0,     0,   420,     0,     0,
      83,     0,    85,     0,     0,     0,     0,   450,     0,     0,
       0,     0,     0,     0,     0,   444,     0,     0,   179,     0,
       0,     0,     0,   269,     0,   315,     0,   331,     0,     0,
       0,     0,     0,     0,     0,     0,   387,     0,     0,     0,
     259,   261,     0,     0,     0,   229,   232,     0,     0,     0,
       0,     0,     0,     0,     0,   252,     0,   206,     0,     0,
       0,     0,   438,   446,     0,    62,   306,     0,     0,   435,
       0,   444,     0,     0,   374,     0,   105,     0,   383,   382,
     348,     0,   103,     0,   361,   346,   347,   366,   364,   381,
     380,    81,     0,     0,     0,    57,    81,    66,   125,   156,
      81,     0,    81,   212,   214,   198,     0,     0,     0,   264,
       0,   263,     0,    30,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   365,   363,   390,     0,     0,   349,
      54,    53,     0,     0,    60,    63,    58,    61,    84,    87,
      86,    68,    70,    67,    69,    93,     0,     0,   124,   123,
       7,   155,   154,   175,     0,     0,   180,   176,   196,   195,
      28,     0,    27,     0,   333,   332,   330,     0,   327,     0,
     210,   429,   208,     0,    23,    21,    22,   221,   220,   228,
       0,   260,   262,   225,   222,     0,   231,   230,     0,   247,
       0,     0,     0,     0,   234,     0,     0,   251,     0,   250,
       0,    26,     0,   207,   211,   211,   101,   100,   440,   439,
     449,     0,     0,   410,   411,     0,   441,     0,     0,   437,
     436,     0,   442,     0,   107,   104,   106,   102,     0,     0,
       0,     0,     0,     0,   216,   218,     0,    81,     0,   202,
     204,     0,   268,   266,     0,     0,     0,   258,   417,     0,
       0,     0,   394,   404,   408,   409,   407,   406,   405,   403,
     402,   401,   400,   399,   397,   434,     0,   373,   372,   371,
     370,   369,   368,   362,   367,   379,   378,   377,   376,   375,
     358,   357,   356,   351,   350,   354,   353,   352,   355,   360,
     359,     0,   445,     0,    51,   451,     0,     0,   174,     0,
       0,   207,   289,   277,     0,     0,     0,   319,   326,     0,
     430,     0,     0,     0,     0,     0,   377,   233,   241,   243,
       0,   244,     0,   242,     0,     0,   249,    19,   254,   255,
       0,   256,   253,   424,     0,    81,    81,   447,   308,   413,
     412,     0,   452,   443,     0,     0,    82,     0,     0,     0,
      73,    72,    77,     0,   128,     0,     0,   126,     0,   138,
       0,   159,     0,     0,   157,     0,     0,   182,   183,    81,
     217,   219,     0,     0,   215,     0,   200,     0,   265,   257,
     267,   418,   416,     0,   392,     0,   433,     0,    31,     0,
       0,    92,    88,    90,   173,   272,     0,   299,     0,   318,
     282,   278,   279,   317,   299,   329,   328,   209,   428,   227,
     226,   224,   223,     0,     0,   235,     0,     0,   236,     0,
       0,    20,   423,     0,     0,     0,     0,     0,   414,     0,
      56,     0,    75,     0,     0,     0,    81,    81,   127,     0,
     151,   149,   147,   148,     0,   145,   142,     0,     0,   158,
       0,   171,   172,     0,   168,     0,     0,   186,   193,   194,
       0,     0,   191,     0,   184,     0,   197,     0,     0,     0,
     199,   203,     0,   393,   398,   432,   431,     0,    52,     0,
       0,   275,   274,   291,     0,     0,     0,     0,     0,   292,
     290,   294,   293,     0,   271,     0,   281,     0,   321,   322,
     324,   323,     0,   320,   245,   246,     0,     0,     0,     0,
     422,   421,   426,   310,     0,     0,   309,     0,   453,    76,
      80,    79,    65,     0,     0,   133,   135,     0,   129,   131,
       0,   122,    81,     0,   139,   164,   166,   160,   162,   170,
     153,   190,     0,   188,     0,     0,   178,   213,   205,     0,
     419,    32,     0,     0,     0,    96,     0,     0,    97,    98,
      99,    91,     0,     0,   295,     0,     0,   302,     0,     0,
       0,   287,   288,     0,   284,   286,   280,     0,   237,   239,
     238,   240,   313,     0,   314,   312,   307,   415,    78,    81,
       0,   150,    81,     0,   146,     0,   144,    81,     0,    81,
       0,   169,   187,   192,     0,   201,     0,   109,     0,     0,
     113,     0,     0,   117,     0,     0,    95,   276,     0,   211,
       0,   301,   303,   300,     0,   270,   283,     0,   316,     0,
       0,   136,     0,   132,     0,   167,     0,   163,   189,   112,
      81,   111,   116,    81,   115,   120,    81,   119,    89,   298,
      81,     0,   304,     0,   285,   311,     0,     0,     0,     0,
     297,   305,     0,     0,     0,     0,   110,   114,   118,   296
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    61,    62,   603,    63,   506,    65,   124,
      66,    67,   221,    68,    69,    70,   226,    71,    72,   509,
     596,   510,   511,   597,   512,   388,    73,    74,    75,   639,
     640,   716,   717,    76,    77,    78,   718,   800,   719,   803,
     720,   806,    79,   228,    80,   390,   517,   752,   753,   749,
     750,   518,   608,   519,   694,   604,   605,    81,   229,    82,
     391,   524,   759,   760,   757,   758,   613,   614,    83,    84,
     230,    85,   526,   527,   528,   529,   621,   622,    86,    87,
      88,   629,    89,   535,    90,   331,   332,   232,   397,   398,
     233,   234,    91,    92,    93,    94,   177,    95,   181,    96,
     184,   185,    97,    98,    99,   240,   241,   100,   321,   462,
     463,   722,   466,   561,   562,   656,   733,   734,   557,   650,
     651,   779,   652,   653,   729,   101,   372,   586,   676,   745,
     102,   323,   467,   564,   663,   103,   159,   327,   328,   104,
     105,   106,   107,   108,   109,   110,   632,   111,   188,   364,
     112,   189,   113,   160,   333,   114,   115,   116,   117,   118,
     194,   371,   138,   203
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -354
static const yytype_int16 yypact[] =
{
    -354,    45,   778,  -354,    56,  -354,  -354,  -354,     9,  -354,
    -354,   143,   191,  3192,    24,   316,  3259,   314,  3326,   110,
    3393,  -354,  -354,  3460,   313,  3527,   399,   403,   505,  -354,
    -354,    75,  3594,   471,   348,  3661,   468,   305,   485,   303,
    -354,  3728,  5121,   149,   194,  -354,  -354,  -354,  5357,  5003,
    5357,   621,  5357,  5357,  5357,  3125,  5357,    48,  5357,  5357,
     552,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  3058,
    -354,  -354,  3058,  -354,  -354,  -354,  -354,  -354,  -354,  -354,
    -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,   161,  3058,
      20,  -354,  -354,  -354,  -354,  -354,  -354,  -354,    71,   302,
    -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  4412,  -354,
    -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,   315,  -354,
    5357,  -354,   294,  -354,   116,   247,   178,  -354,  4180,   342,
    -354,   367,  -354,   394,   182,  4241,   400,  -354,   -28,   404,
    4469,   429,   455,  4518,   456,  5818,    99,   459,  -354,  3058,
     472,  4575,   482,  -354,   486,  -354,   494,  -354,  4624,   499,
      96,   522,   525,   531,   544,  5818,   559,   560,   491,   343,
    -354,  -354,   568,  4681,   573,  -354,  -354,   123,   580,   100,
     166,   120,   583,   508,   210,  -354,   594,  -354,    31,    31,
     596,  4730,  -354,  5818,   108,  -354,  -354,  5567,  5159,  -354,
     546,  5413,    62,    64,  6063,   600,  -354,   214,   640,   640,
     340,   601,  -354,   221,   340,  -354,  -354,   340,   340,  -354,
    -354,  -354,   567,   613,   297,  -354,  -354,  -354,  -354,  -354,
    -354,   480,  -354,  -354,  -354,  -354,   620,   222,   627,  -354,
     223,   498,   228,  -354,  5239,  5041,   610,  5357,  5357,  5357,
    5357,  5357,  5357,  5357,  5357,  5357,  5357,  5357,  5357,  3795,
    5357,  5357,  5357,  5357,  5357,  5357,  5357,  5357,   626,  5357,
    5357,  5357,  5357,  5357,  5357,  5357,  5357,  5357,  5357,  5357,
    5357,  5357,  5357,  5357,  -354,  -354,  -354,  5357,  5357,  5818,
    -354,  -354,   628,  5357,  -354,  -354,  -354,  -354,  -354,  -354,
    -354,  -354,  -354,  -354,  -354,  -354,   628,  3862,  -354,  -354,
    -354,  -354,  -354,  -354,   633,  5357,  -354,  -354,  -354,  -354,
    -354,   289,  -354,   454,  -354,  -354,  -354,   253,  -354,   636,
    -354,   562,  -354,   582,  -354,  -354,  -354,  -354,  -354,  -354,
     460,  -354,  -354,  -354,  -354,  3929,  -354,  -354,   638,  -354,
     643,    16,    23,   656,  -354,   329,   654,  -354,    54,  -354,
     655,  -354,   661,   659,   161,   161,  -354,  -354,  -354,  -354,
    -354,  5357,   663,  -354,  -354,  5616,  -354,  5277,  5357,  -354,
    -354,   607,  -354,  5357,  -354,  -354,  -354,  -354,  1804,  1462,
     479,   584,  1576,   419,  -354,  -354,  1918,  -354,  3058,  -354,
    -354,   151,  -354,  -354,   662,   667,   274,  -354,  -354,   226,
    5357,  5462,  -354,  5818,  5818,  5818,  5818,  5818,  5818,  5818,
    5818,  5818,  5818,  5818,  5867,  -354,  4119,  6014,  6063,  4036,
    4036,  4036,  4036,  4036,  4036,  -354,   640,   640,   640,   640,
     550,   550,   417,   581,   581,   736,   736,   736,   251,   340,
     340,  4296,  5916,   602,  5818,  -354,   669,  4357,  -354,   275,
     676,   659,  -354,   648,   686,   684,   689,  -354,  -354,   553,
    -354,   659,  5357,   693,   694,   695,   280,  -354,  -354,  -354,
     697,  -354,   699,  -354,   291,   296,  -354,  -354,  -354,  -354,
     696,  -354,  -354,  -354,   271,  -354,  -354,  5818,  -354,  -354,
    -354,  5511,  5818,  -354,  5671,   703,  -354,   181,  3996,   698,
    -354,  -354,  -354,   705,  -354,    27,   489,  -354,   700,  -354,
     707,  -354,    92,   704,  -354,    65,   709,   682,  -354,  -354,
    -354,  -354,   711,  2032,  -354,   664,  -354,   453,  -354,  -354,
    -354,  -354,  -354,  5720,  -354,  5357,  -354,  4063,  -354,  5357,
    5357,  -354,  -354,  -354,  -354,  -354,   273,    80,   716,  -354,
     665,   641,  -354,  -354,    82,  -354,  -354,  -354,  5818,  -354,
    -354,  -354,  -354,   720,   722,  -354,   721,   724,  -354,   726,
     730,  -354,  -354,   723,  2146,  2260,   334,  5357,  -354,  5357,
    -354,   725,  -354,   746,  4787,   747,  -354,  -354,  -354,   467,
    -354,  -354,  -354,   677,    59,  -354,  -354,   751,   543,  -354,
     549,  -354,  -354,   114,  -354,   754,   756,  -354,  -354,  -354,
     628,    12,  -354,   757,  -354,  1690,  -354,   758,   715,   706,
    -354,  -354,   712,  -354,   688,  -354,  5965,   276,  5818,   892,
    3058,  -354,  -354,  -354,   687,   762,   765,   767,    11,  -354,
    -354,  -354,  -354,   768,  -354,   320,  -354,   684,  -354,  -354,
    -354,  -354,   771,  -354,  -354,  -354,   773,   789,   791,   796,
    -354,  -354,  -354,  -354,   104,   799,  -354,  5769,  5818,  -354,
    -354,  -354,  -354,  2374,  1462,  -354,  -354,    10,  -354,  -354,
      49,  -354,  -354,  3058,  -354,  -354,  -354,  -354,  -354,   575,
    -354,  -354,   808,  -354,   576,   628,  -354,  -354,  -354,   813,
    -354,  -354,   492,   542,   547,  -354,   816,   892,  -354,  -354,
    -354,  -354,   769,  5357,  -354,   760,   835,  -354,   834,   282,
     841,  -354,  -354,    86,  -354,  -354,  -354,   842,  -354,  -354,
    -354,  -354,  -354,   565,  -354,  -354,  -354,  -354,  -354,  -354,
    3058,  -354,  -354,  3058,  -354,  2488,  -354,  -354,  3058,  -354,
    3058,  -354,  -354,  -354,   843,  -354,   844,  -354,  3058,   847,
    -354,  3058,   848,  -354,  3058,   849,  -354,  -354,  4836,   161,
    5357,  -354,  -354,  -354,    83,  -354,  -354,   320,  -354,   288,
    1006,  -354,  1120,  -354,  1234,  -354,  1348,  -354,  -354,  -354,
    -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,
    -354,  4893,  -354,   850,  -354,  -354,  2602,  2716,  2830,  2944,
    -354,  -354,   851,   852,   854,   855,  -354,  -354,  -354,  -354
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -354,  -354,  -354,  -354,  -354,  -353,  -354,    -2,  -354,  -354,
    -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,    90,
    -354,  -354,  -354,  -354,  -354,  -208,  -354,  -354,  -354,  -354,
    -354,   136,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,
    -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,
    -354,   469,  -354,  -354,  -354,  -354,   169,  -354,  -354,  -354,
    -354,  -354,  -354,  -354,  -354,  -354,  -354,   163,  -354,  -354,
    -354,  -354,  -354,  -354,   336,  -354,  -354,   160,  -354,  -345,
    -354,  -354,  -354,  -354,  -354,  -227,   395,  -316,  -354,  -354,
    -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,
    -354,   507,  -354,  -354,  -354,   -90,  -354,  -354,  -354,  -354,
    -354,  -354,   405,  -354,   212,  -354,  -354,    84,  -354,  -354,
     306,  -354,   308,   311,  -354,  -354,  -354,  -354,  -354,    87,
    -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,   408,  -354,
    -337,   -10,  -354,   -12,    -3,   853,  -354,  -354,  -354,   690,
    -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,  -354,    29,
    -354,  -354,  -354,  -354
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -449
static const yytype_int16 yytable[] =
{
      64,   128,   125,   474,   135,   491,   140,   137,   143,   242,
     401,   145,   726,   151,   487,   703,   158,   727,   389,   479,
     165,   236,   392,   173,   396,   129,   481,   130,   599,   191,
     193,   487,   362,   600,   601,   602,   197,   201,   204,   145,
     208,   209,   210,   145,   214,     3,   217,   218,   495,   496,
     220,   306,   146,   487,   215,   600,   601,   602,   487,   119,
     488,   489,   688,   379,   307,   381,   616,   225,   617,   618,
     227,   619,   238,   120,  -263,   216,   161,   239,   202,   237,
     207,   162,   163,   643,   213,   658,   644,   235,   644,   812,
     363,   704,   480,   610,   728,  -170,   611,   329,   612,   482,
     314,   350,   330,  -248,   705,   286,   689,   742,   289,   368,
     490,   141,  -448,  -448,  -448,  -448,  -448,   697,   645,   291,
     645,   353,   380,   354,   382,   286,   347,   490,   646,   647,
     646,   647,   286,  -248,  -448,  -448,   494,   286,   690,  -170,
     286,   288,   286,   383,   786,  -448,   121,   317,   286,   490,
    -263,   406,   536,   355,   490,   286,  -448,   620,  -448,  -207,
    -448,   698,   286,  -448,  -448,   787,   813,  -448,   369,  -448,
     286,  -170,   351,   352,   648,  -207,   648,   315,   288,  -248,
     743,   294,   591,   744,   592,   301,   375,   370,   286,   533,
     286,  -448,   122,   699,   286,   292,   195,   123,   286,   356,
     196,   286,   348,  -448,  -448,   286,   286,   286,  -448,   537,
     231,   286,   649,   359,   286,   286,  -448,   385,  -448,   659,
    -448,  -448,  -448,   400,   387,   295,   403,   541,   330,   302,
     471,   407,   145,   411,   556,   413,   414,   415,   416,   417,
     418,   419,   420,   421,   422,   423,   424,   426,   427,   428,
     429,   430,   431,   432,   433,   434,   468,   436,   437,   438,
     439,   440,   441,   442,   443,   444,   445,   446,   447,   448,
     449,   450,   582,   409,   641,   451,   452,   540,   554,   711,
    -207,   454,   453,   572,   542,   783,   286,   584,   585,   360,
     460,   742,  -273,   288,   575,   457,   455,   290,   186,   578,
     288,  -207,   404,   145,   186,   288,   178,   404,   239,   187,
     244,   179,   245,   246,   147,   136,   148,   131,   735,   132,
     123,   625,  -273,   293,     6,     7,   731,     9,    10,   583,
     133,   642,   469,   476,   751,   484,   485,   673,   180,   244,
     674,   245,   246,   675,   459,   298,   341,   732,   461,   169,
     471,   170,   471,   404,   288,   288,  -420,   282,   283,   497,
     149,   784,  -420,   284,   285,   501,   502,   744,   576,   577,
     299,   504,   286,   579,   580,    45,    46,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   683,   684,
     342,   287,   284,   285,   288,   171,   534,   300,   543,   244,
     152,   245,   246,   305,   154,   153,   789,   308,   286,   155,
     286,   286,   286,   286,   286,   286,   286,   286,   286,   286,
     286,   286,   530,   286,   286,   286,   286,   286,   286,   286,
     286,   286,   310,   286,   286,   286,   286,   286,   286,   286,
     286,   286,   286,   286,   286,   286,   286,   286,   286,   286,
     735,   286,   284,   285,   286,   464,   630,  -277,   311,   313,
     568,   473,   316,   810,     6,     7,   531,     9,    10,   174,
     685,   175,   167,   286,   176,   318,   244,   168,   245,   246,
     513,   393,   514,   394,   755,   320,   182,   465,  -137,   322,
     606,   183,  -141,   766,   286,   767,   594,   324,   286,   286,
     631,   286,   515,   516,   239,   326,   156,   405,   157,     6,
       7,     8,     9,    10,   686,    45,    46,   276,   277,   278,
     279,   280,   281,   282,   283,   334,  -140,   395,   335,   284,
     285,    21,    22,   145,   336,   636,  -141,   145,   638,   768,
     286,   790,    30,   769,   792,   770,   692,   337,   772,   794,
     773,   796,   695,   127,   565,    40,   219,    42,   123,   326,
      45,    46,   338,   339,    48,   286,    49,   340,   152,     6,
       7,   343,     9,    10,   634,   677,   346,   678,   637,   611,
     618,   612,   619,   349,   358,   520,   357,   521,    50,   771,
     693,   286,   816,  -137,   774,   817,   696,   361,   818,   366,
      52,    53,   819,   384,   386,    54,   376,   522,   516,   244,
     702,   245,   246,    56,   154,    57,   412,    58,    59,    60,
      45,    46,   205,   399,   206,     6,     7,     8,     9,    10,
     402,  -140,   435,   286,   123,   286,   458,   715,   721,   470,
     244,   471,   245,   246,   477,   472,   478,    21,    22,   275,
     276,   277,   278,   279,   280,   281,   282,   283,    30,   483,
     486,   183,   284,   285,   493,   330,   498,   503,   538,   127,
     539,    40,   551,    42,   286,   286,    45,    46,   550,   555,
      48,   465,    49,   278,   279,   280,   281,   282,   283,   559,
     560,   756,   563,   284,   285,   764,   569,   570,   571,   244,
     581,   245,   246,   573,    50,   574,   590,   595,   598,   607,
     609,   778,   525,   615,   626,   715,    52,    53,   623,   654,
     657,    54,   628,   664,   655,   665,   670,   666,   679,    56,
     667,    57,   668,    58,    59,    60,   669,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   791,   680,
     682,   793,   284,   285,   691,   687,   795,   700,   797,   701,
     706,   707,   708,   723,   709,   724,   801,   288,   811,   804,
     710,   187,   807,   725,   748,   286,   738,   730,    -2,     4,
     737,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,   739,    16,   740,   244,    17,   245,   246,   741,
      18,    19,   746,    20,    21,    22,    23,    24,   286,    25,
      26,   762,    27,    28,    29,    30,   765,    31,    32,    33,
      34,    35,    36,    37,    38,   775,    39,   777,    40,    41,
      42,    43,    44,    45,    46,    47,   780,    48,   781,    49,
     782,   281,   282,   283,   785,   788,   798,   799,   284,   285,
     802,   805,   808,   776,   826,   827,   821,   828,   829,   754,
     523,    50,   761,   624,   763,    51,   567,   492,   558,   736,
     660,   814,   661,    52,    53,   662,   815,   566,    54,   365,
       0,     0,     0,     0,    55,   166,    56,     0,    57,     0,
      58,    59,    60,     4,     0,     5,     6,     7,     8,     9,
      10,   -94,    12,    13,    14,    15,     0,    16,     0,     0,
      17,   712,   713,   714,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   222,     0,   223,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     224,     0,    40,    41,    42,    43,     0,    45,    46,    47,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
       0,     0,    54,     0,     0,     0,     0,     0,    55,     0,
      56,     0,    57,     0,    58,    59,    60,     4,     0,     5,
       6,     7,     8,     9,    10,  -134,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,  -134,
    -134,    20,    21,    22,    23,    24,     0,    25,   222,     0,
     223,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,  -134,   224,     0,    40,    41,    42,    43,
       0,    45,    46,    47,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,     0,
       0,     0,    55,     0,    56,     0,    57,     0,    58,    59,
      60,     4,     0,     5,     6,     7,     8,     9,    10,  -130,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,  -130,  -130,    20,    21,    22,    23,    24,
       0,    25,   222,     0,   223,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,  -130,   224,     0,
      40,    41,    42,    43,     0,    45,    46,    47,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
      54,     0,     0,     0,     0,     0,    55,     0,    56,     0,
      57,     0,    58,    59,    60,     4,     0,     5,     6,     7,
       8,     9,    10,  -165,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,  -165,  -165,    20,
      21,    22,    23,    24,     0,    25,   222,     0,   223,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,  -165,   224,     0,    40,    41,    42,    43,     0,    45,
      46,    47,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,     0,
      55,     0,    56,     0,    57,     0,    58,    59,    60,     4,
       0,     5,     6,     7,     8,     9,    10,  -161,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,  -161,  -161,    20,    21,    22,    23,    24,     0,    25,
     222,     0,   223,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,  -161,   224,     0,    40,    41,
      42,    43,     0,    45,    46,    47,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,    51,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,     0,     0,    54,     0,
       0,     0,     0,     0,    55,     0,    56,     0,    57,     0,
      58,    59,    60,     4,     0,     5,     6,     7,     8,     9,
      10,   -71,    12,    13,    14,    15,     0,    16,   507,   508,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   222,     0,   223,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     224,     0,    40,    41,    42,    43,     0,    45,    46,    47,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
       0,     0,    54,     0,     0,     0,     0,     0,    55,     0,
      56,     0,    57,     0,    58,    59,    60,     4,     0,     5,
       6,     7,     8,     9,    10,  -181,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,   525,    25,   222,     0,
     223,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   224,     0,    40,    41,    42,    43,
       0,    45,    46,    47,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,     0,
       0,     0,    55,     0,    56,     0,    57,     0,    58,    59,
      60,     4,     0,     5,     6,     7,     8,     9,    10,  -185,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
    -185,    25,   222,     0,   223,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   224,     0,
      40,    41,    42,    43,     0,    45,    46,    47,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
      54,     0,     0,     0,     0,     0,    55,     0,    56,     0,
      57,     0,    58,    59,    60,     4,     0,     5,     6,     7,
       8,     9,    10,   505,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   222,     0,   223,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   224,     0,    40,    41,    42,    43,     0,    45,
      46,    47,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,     0,
      55,     0,    56,     0,    57,     0,    58,    59,    60,     4,
       0,     5,     6,     7,     8,     9,    10,   532,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     222,     0,   223,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   224,     0,    40,    41,
      42,    43,     0,    45,    46,    47,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,    51,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,     0,     0,    54,     0,
       0,     0,     0,     0,    55,     0,    56,     0,    57,     0,
      58,    59,    60,     4,     0,     5,     6,     7,     8,     9,
      10,   627,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   222,     0,   223,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     224,     0,    40,    41,    42,    43,     0,    45,    46,    47,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
       0,     0,    54,     0,     0,     0,     0,     0,    55,     0,
      56,     0,    57,     0,    58,    59,    60,     4,     0,     5,
       6,     7,     8,     9,    10,   671,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   222,     0,
     223,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   224,     0,    40,    41,    42,    43,
       0,    45,    46,    47,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,     0,
       0,     0,    55,     0,    56,     0,    57,     0,    58,    59,
      60,     4,     0,     5,     6,     7,     8,     9,    10,   672,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   222,     0,   223,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   224,     0,
      40,    41,    42,    43,     0,    45,    46,    47,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
      54,     0,     0,     0,     0,     0,    55,     0,    56,     0,
      57,     0,    58,    59,    60,     4,     0,     5,     6,     7,
       8,     9,    10,   -74,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   222,     0,   223,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   224,     0,    40,    41,    42,    43,     0,    45,
      46,    47,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,     0,
      55,     0,    56,     0,    57,     0,    58,    59,    60,     4,
       0,     5,     6,     7,     8,     9,    10,  -143,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     222,     0,   223,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   224,     0,    40,    41,
      42,    43,     0,    45,    46,    47,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,    51,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,     0,     0,    54,     0,
       0,     0,     0,     0,    55,     0,    56,     0,    57,     0,
      58,    59,    60,     4,     0,     5,     6,     7,     8,     9,
      10,   822,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   222,     0,   223,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     224,     0,    40,    41,    42,    43,     0,    45,    46,    47,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
       0,     0,    54,     0,     0,     0,     0,     0,    55,     0,
      56,     0,    57,     0,    58,    59,    60,     4,     0,     5,
       6,     7,     8,     9,    10,   823,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   222,     0,
     223,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   224,     0,    40,    41,    42,    43,
       0,    45,    46,    47,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,     0,
       0,     0,    55,     0,    56,     0,    57,     0,    58,    59,
      60,     4,     0,     5,     6,     7,     8,     9,    10,   824,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   222,     0,   223,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   224,     0,
      40,    41,    42,    43,     0,    45,    46,    47,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
      54,     0,     0,     0,     0,     0,    55,     0,    56,     0,
      57,     0,    58,    59,    60,     4,     0,     5,     6,     7,
       8,     9,    10,   825,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   222,     0,   223,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   224,     0,    40,    41,    42,    43,     0,    45,
      46,    47,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,     0,
      55,     0,    56,     0,    57,     0,    58,    59,    60,     4,
       0,     5,     6,     7,     8,     9,    10,     0,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     222,     0,   223,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   224,     0,    40,    41,
      42,    43,     0,    45,    46,    47,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,   211,     0,   212,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,    51,     0,     0,     0,     0,
       0,    21,    22,    52,    53,     0,     0,     0,    54,     0,
       0,     0,    30,     0,    55,     0,    56,     0,    57,     0,
      58,    59,    60,   127,     0,    40,     0,    42,     0,     0,
      45,    46,     0,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,   126,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
      52,    53,     0,     0,     0,    54,     0,     0,     0,    30,
       0,     0,     0,    56,     0,    57,     0,    58,    59,    60,
     127,     0,    40,     0,    42,     0,     0,    45,    46,     0,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
     134,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,    52,    53,     0,
       0,     0,    54,     0,     0,     0,    30,     0,     0,     0,
      56,     0,    57,     0,    58,    59,    60,   127,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,     0,
      49,     0,     0,     0,     0,     0,     0,   139,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,    57,
       0,    58,    59,    60,   127,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,   142,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,    52,    53,     0,     0,     0,    54,     0,     0,     0,
      30,     0,     0,     0,    56,     0,    57,     0,    58,    59,
      60,   127,     0,    40,     0,    42,     0,     0,    45,    46,
       0,     0,    48,     0,    49,     0,     0,     0,     0,     0,
       0,   144,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,    52,    53,
       0,     0,     0,    54,     0,     0,     0,    30,     0,     0,
       0,    56,     0,    57,     0,    58,    59,    60,   127,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,   150,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    21,    22,    52,    53,     0,     0,     0,
      54,     0,     0,     0,    30,     0,     0,     0,    56,     0,
      57,     0,    58,    59,    60,   127,     0,    40,     0,    42,
       0,     0,    45,    46,     0,     0,    48,     0,    49,     0,
       0,     0,     0,     0,     0,   164,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,    57,     0,    58,
      59,    60,   127,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,   172,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,    52,
      53,     0,     0,     0,    54,     0,     0,     0,    30,     0,
       0,     0,    56,     0,    57,     0,    58,    59,    60,   127,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,     0,     0,     0,     0,   190,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,    52,    53,     0,     0,
       0,    54,     0,     0,     0,    30,     0,     0,     0,    56,
       0,    57,     0,    58,    59,    60,   127,     0,    40,     0,
      42,     0,     0,    45,    46,     0,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,   425,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,    52,    53,     0,     0,     0,    54,     0,
       0,     0,    30,     0,     0,     0,    56,     0,    57,     0,
      58,    59,    60,   127,     0,    40,     0,    42,     0,     0,
      45,    46,     0,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,   456,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
      52,    53,     0,     0,     0,    54,     0,     0,     0,    30,
       0,     0,     0,    56,     0,    57,     0,    58,    59,    60,
     127,     0,    40,     0,    42,     0,     0,    45,    46,     0,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
     475,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,    52,    53,     0,
       0,     0,    54,     0,     0,     0,    30,     0,     0,     0,
      56,     0,    57,     0,    58,    59,    60,   127,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,     0,
      49,     0,     0,     0,     0,     0,     0,   593,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,    57,
       0,    58,    59,    60,   127,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,   635,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,    52,    53,     0,     0,   244,    54,   245,   246,     0,
      30,     0,     0,     0,    56,     0,    57,     0,    58,    59,
      60,   127,     0,    40,     0,    42,     0,     0,    45,    46,
     546,     0,    48,     0,    49,     0,   268,   269,   270,   271,
     272,     0,     0,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,     0,     0,    50,     0,   284,   285,
       0,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,   547,     0,     0,     0,
       0,    56,     0,    57,     0,    58,    59,    60,   244,     0,
     245,   246,     0,   296,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,     0,     0,     0,   259,
     260,   261,     0,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,     0,     0,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   297,     0,     0,
       0,   284,   285,     0,     0,     0,     0,     0,     0,   244,
       0,   245,   246,     0,   303,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,     0,     0,     0,
     259,   260,   261,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,     0,     0,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   304,     0,
       0,     0,   284,   285,     0,     0,     0,     0,     0,   548,
     244,     0,   245,   246,     0,     0,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,     0,     0,
       0,   259,   260,   261,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,     0,     0,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,     0,
       0,     0,     0,   284,   285,   244,     0,   245,   246,     0,
     552,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,     0,     0,   549,   259,   260,   261,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,     0,     0,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   553,     0,     0,     0,   284,   285,
       0,     0,     0,     0,     0,   243,   244,     0,   245,   246,
       0,     0,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,     0,     0,     0,   259,   260,   261,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,     0,     0,     0,     0,   284,
     285,   244,   309,   245,   246,     0,     0,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   257,   258,     0,
       0,     0,   259,   260,   261,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,     0,     0,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
       0,   312,     0,     0,   284,   285,     0,     0,   244,     0,
     245,   246,     0,     0,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,     0,     0,     0,   259,
     260,   261,     0,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,     0,     0,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   244,   319,   245,
     246,   284,   285,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,     0,     0,     0,   259,   260,
     261,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,     0,     0,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   283,     0,   325,     0,     0,
     284,   285,     0,     0,   244,     0,   245,   246,     0,     0,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     257,   258,     0,     0,     0,   259,   260,   261,     0,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   244,   344,   245,   246,   284,   285,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,   257,
     258,     0,     0,     0,   259,   260,   261,     0,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,     0,
       0,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,     0,   367,     0,     0,   284,   285,     0,     0,
     244,     0,   245,   246,     0,     0,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,     0,     0,
       0,   259,   260,   261,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   345,   271,   272,     0,     0,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   244,
     681,   245,   246,   284,   285,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,     0,     0,     0,
     259,   260,   261,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,     0,     0,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,     0,   809,
       0,     0,   284,   285,     0,     0,   244,     0,   245,   246,
       0,     0,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,     0,     0,     0,   259,   260,   261,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   244,   820,   245,   246,   284,
     285,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,     0,     0,     0,   259,   260,   261,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,     0,     0,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,     0,     0,     0,     0,   284,   285,
       0,     0,   244,     0,   245,   246,     0,     0,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
       0,     0,     0,   259,   260,   261,     0,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,     0,     0,     0,     0,   284,   285,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      30,     0,     0,     0,     0,     6,     7,     8,     9,    10,
     198,   127,     0,    40,     0,    42,     0,     0,    45,    46,
       0,     0,    48,   199,    49,     0,   200,    21,    22,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    30,     0,
       0,     0,     0,     0,     0,     0,    50,     0,   198,   127,
       0,    40,     0,    42,     0,     0,    45,    46,    52,    53,
      48,     0,    49,    54,     0,     0,     0,     0,     0,     0,
       0,    56,     0,    57,     0,    58,    59,    60,     0,     0,
       0,     0,     0,     0,    50,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,    52,    53,     0,     0,
       0,    54,     0,     0,     0,   410,     0,    21,    22,    56,
       0,    57,     0,    58,    59,    60,     0,     0,    30,     0,
       0,     0,     0,     6,     7,     8,     9,    10,     0,   127,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,   192,    49,     0,     0,    21,    22,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    30,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,   127,     0,    40,
       0,    42,     0,     0,    45,    46,    52,    53,    48,   374,
      49,    54,     0,     0,     0,     0,     0,     0,     0,    56,
       0,    57,     0,    58,    59,    60,     0,     0,     0,     0,
       0,     0,    50,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,    52,    53,     0,     0,     0,    54,
       0,     0,     0,     0,     0,    21,    22,    56,     0,    57,
       0,    58,    59,    60,     0,     0,    30,     0,     0,     0,
       0,     6,     7,     8,     9,    10,     0,   127,     0,    40,
       0,    42,     0,     0,    45,    46,     0,   408,    48,     0,
      49,     0,     0,    21,    22,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    30,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,   127,     0,    40,     0,    42,
       0,     0,    45,    46,    52,    53,    48,   500,    49,    54,
       0,     0,     0,     0,     0,     0,     0,    56,     0,    57,
       0,    58,    59,    60,     0,     0,     0,     0,     0,     0,
      50,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,     0,     0,
       0,     0,     0,    21,    22,    56,     0,    57,     0,    58,
      59,    60,     0,     0,    30,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   127,     0,    40,     0,    42,
       0,     0,    45,    46,     0,     0,    48,     0,    49,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,     0,     0,
     377,     0,     0,     0,     0,    56,     0,    57,     0,    58,
      59,    60,   244,     0,   245,   246,   378,     0,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
       0,     0,     0,   259,   260,   261,     0,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,   377,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   244,   544,   245,   246,   284,   285,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   257,   258,     0,
       0,     0,   259,   260,   261,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,     0,   587,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     244,   588,   245,   246,   284,   285,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,     0,     0,
       0,   259,   260,   261,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,     0,     0,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,     0,
       0,     0,     0,   284,   285,   373,   244,     0,   245,   246,
       0,     0,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,     0,     0,     0,   259,   260,   261,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   244,   499,   245,   246,   284,
     285,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,     0,     0,     0,   259,   260,   261,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,     0,     0,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,     0,     0,     0,     0,   284,   285,
     244,     0,   245,   246,   589,     0,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,     0,     0,
       0,   259,   260,   261,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,     0,     0,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   244,
     633,   245,   246,   284,   285,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,     0,     0,     0,
     259,   260,   261,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,     0,     0,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   244,   747,
     245,   246,   284,   285,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,     0,     0,     0,   259,
     260,   261,     0,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,     0,     0,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   244,     0,   245,
     246,   284,   285,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,     0,     0,     0,   259,   260,
     261,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,     0,     0,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   283,   244,     0,   245,   246,
     284,   285,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   258,     0,     0,   545,   259,   260,   261,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   244,     0,   245,   246,   284,
     285,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   259,   260,   261,     0,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,     0,     0,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   244,     0,   245,   246,   284,   285,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   260,   261,     0,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   244,     0,   245,   246,   284,   285,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   261,     0,   262,   263,
     264,   265,   266,   267,   268,   269,   270,   271,   272,     0,
       0,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   244,     0,   245,   246,   284,   285,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,     0,     0,     0,     0,   284,   285
};

static const yytype_int16 yycheck[] =
{
       2,    13,    12,   340,    16,   358,    18,    17,    20,    99,
     237,    23,     1,    25,     4,     3,    28,     6,   226,     3,
      32,     1,   230,    35,   232,     1,     3,     3,     1,    41,
      42,     4,     1,     6,     7,     8,    48,    49,    50,    51,
      52,    53,    54,    55,    56,     0,    58,    59,   364,   365,
      60,    79,    23,     4,     6,     6,     7,     8,     4,     3,
       6,     7,     3,     1,    92,     1,     1,    69,     3,     4,
      72,     6,     1,    64,     3,    27,     1,     6,    49,    59,
      51,     6,     7,     3,    55,     3,     6,    89,     6,     6,
      59,    79,    76,     1,    83,     3,     4,     1,     6,    76,
       1,     1,     6,     3,    92,   108,    47,     3,   120,     1,
     100,     1,     4,     5,     6,     7,     8,     3,    38,     3,
      38,     1,    60,     3,    60,   128,     3,   100,    48,    49,
      48,    49,   135,    33,    26,    27,   363,   140,    79,    47,
     143,    79,   145,    79,    58,    37,     3,   149,   151,   100,
      79,   241,     1,    33,   100,   158,    48,    92,    50,    63,
      52,    47,   165,    55,    56,    79,    83,    59,    60,    61,
     173,    79,     6,     7,    94,    79,    94,    78,    79,    79,
      76,     3,     1,    79,     3,     3,   198,    79,   191,   397,
     193,    83,     1,    79,   197,    79,    47,     6,   201,    79,
       6,   204,    79,    95,    96,   208,   209,   210,   100,    58,
      49,   214,   557,     3,   217,   218,   108,     3,   110,   564,
     112,   113,   114,     1,     3,    47,     3,     1,     6,    47,
      79,     3,   244,   245,   461,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,     3,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,     1,   244,     1,   287,   288,     3,     3,     3,
      58,   293,   292,     3,    58,     3,   289,   495,   496,    79,
       1,     3,     3,    79,     3,   307,   306,     3,     1,     3,
      79,    79,    79,   315,     1,    79,     1,    79,     6,     6,
      59,     6,    61,    62,     1,     1,     3,     1,   655,     3,
       6,   529,    33,    76,     4,     5,     6,     7,     8,    58,
      14,    58,    79,   345,   687,     6,     7,     3,    33,    59,
       6,    61,    62,     9,   315,     3,     3,    27,    59,     1,
      79,     3,    79,    79,    79,    79,    59,   106,   107,   371,
      47,    79,    59,   112,   113,   377,   378,    79,    77,    78,
       3,   383,   375,    77,    78,    55,    56,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,   107,   596,   597,
      47,    76,   112,   113,    79,    47,   398,     3,   410,    59,
       1,    61,    62,     3,     1,     6,   743,     3,   411,     6,
     413,   414,   415,   416,   417,   418,   419,   420,   421,   422,
     423,   424,     3,   426,   427,   428,   429,   430,   431,   432,
     433,   434,     3,   436,   437,   438,   439,   440,   441,   442,
     443,   444,   445,   446,   447,   448,   449,   450,   451,   452,
     787,   454,   112,   113,   457,     1,     3,     3,     3,     3,
     472,     1,     3,   779,     4,     5,    47,     7,     8,     1,
       3,     3,     1,   476,     6,     3,    59,     6,    61,    62,
       1,     1,     3,     3,   692,     3,     1,    33,     9,     3,
       1,     6,     3,     1,   497,     3,   508,     3,   501,   502,
      47,   504,    23,    24,     6,     6,     1,     9,     3,     4,
       5,     6,     7,     8,    47,    55,    56,   100,   101,   102,
     103,   104,   105,   106,   107,     3,    47,    47,     3,   112,
     113,    26,    27,   545,     3,   547,    47,   549,   550,    47,
     543,   749,    37,     1,   752,     3,     3,     3,     1,   757,
       3,   759,     3,    48,     1,    50,     4,    52,     6,     6,
      55,    56,     3,     3,    59,   568,    61,    76,     1,     4,
       5,     3,     7,     8,   545,   587,     3,   589,   549,     4,
       4,     6,     6,     3,    76,     1,     3,     3,    83,    47,
      47,   594,   800,     9,    47,   803,    47,     3,   806,     3,
      95,    96,   810,     3,     3,   100,    60,    23,    24,    59,
     620,    61,    62,   108,     1,   110,     6,   112,   113,   114,
      55,    56,     1,     3,     3,     4,     5,     6,     7,     8,
       3,    47,     6,   636,     6,   638,     3,   639,   640,     3,
      59,    79,    61,    62,     6,    63,     3,    26,    27,    99,
     100,   101,   102,   103,   104,   105,   106,   107,    37,     3,
       6,     6,   112,   113,     3,     6,     3,    60,     6,    48,
       3,    50,     3,    52,   677,   678,    55,    56,    76,     3,
      59,    33,    61,   102,   103,   104,   105,   106,   107,     3,
       6,   693,     3,   112,   113,   705,     3,     3,     3,    59,
       4,    61,    62,     6,    83,     6,     3,     9,     3,     9,
       3,   723,    30,     9,     3,   717,    95,    96,     9,     3,
      79,   100,    58,     3,    59,     3,     3,     6,     3,   108,
       6,   110,     6,   112,   113,   114,     6,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,   107,   750,     3,
       3,   753,   112,   113,     3,    78,   758,     3,   760,     3,
       3,     3,    47,    76,    58,     3,   768,    79,   780,   771,
      58,     6,   774,     6,   684,   778,     3,     9,     0,     1,
       9,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,     3,    15,     3,    59,    18,    61,    62,     3,
      22,    23,     3,    25,    26,    27,    28,    29,   811,    31,
      32,     3,    34,    35,    36,    37,     3,    39,    40,    41,
      42,    43,    44,    45,    46,     9,    48,    58,    50,    51,
      52,    53,    54,    55,    56,    57,    76,    59,     3,    61,
       6,   105,   106,   107,     3,     3,     3,     3,   112,   113,
       3,     3,     3,   717,     3,     3,     6,     3,     3,   690,
     391,    83,   699,   527,   704,    87,   471,   360,   463,   657,
     564,   787,   564,    95,    96,   564,   789,   469,   100,   189,
      -1,    -1,    -1,    -1,   106,    32,   108,    -1,   110,    -1,
     112,   113,   114,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    19,    20,    21,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    -1,    55,    56,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    95,    96,    -1,
      -1,    -1,   100,    -1,    -1,    -1,    -1,    -1,   106,    -1,
     108,    -1,   110,    -1,   112,   113,   114,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,
      24,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    47,    48,    -1,    50,    51,    52,    53,
      -1,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    95,    96,    -1,    -1,    -1,   100,    -1,    -1,    -1,
      -1,    -1,   106,    -1,   108,    -1,   110,    -1,   112,   113,
     114,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    47,    48,    -1,
      50,    51,    52,    53,    -1,    55,    56,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    -1,    -1,    -1,    87,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    95,    96,    -1,    -1,    -1,
     100,    -1,    -1,    -1,    -1,    -1,   106,    -1,   108,    -1,
     110,    -1,   112,   113,   114,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    47,    48,    -1,    50,    51,    52,    53,    -1,    55,
      56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,
      -1,    87,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    95,
      96,    -1,    -1,    -1,   100,    -1,    -1,    -1,    -1,    -1,
     106,    -1,   108,    -1,   110,    -1,   112,   113,   114,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    23,    24,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    47,    48,    -1,    50,    51,
      52,    53,    -1,    55,    56,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    95,    96,    -1,    -1,    -1,   100,    -1,
      -1,    -1,    -1,    -1,   106,    -1,   108,    -1,   110,    -1,
     112,   113,   114,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    16,    17,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    -1,    55,    56,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    95,    96,    -1,
      -1,    -1,   100,    -1,    -1,    -1,    -1,    -1,   106,    -1,
     108,    -1,   110,    -1,   112,   113,   114,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    30,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      -1,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    95,    96,    -1,    -1,    -1,   100,    -1,    -1,    -1,
      -1,    -1,   106,    -1,   108,    -1,   110,    -1,   112,   113,
     114,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      30,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    -1,    55,    56,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    -1,    -1,    -1,    87,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    95,    96,    -1,    -1,    -1,
     100,    -1,    -1,    -1,    -1,    -1,   106,    -1,   108,    -1,
     110,    -1,   112,   113,   114,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    -1,    55,
      56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,
      -1,    87,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    95,
      96,    -1,    -1,    -1,   100,    -1,    -1,    -1,    -1,    -1,
     106,    -1,   108,    -1,   110,    -1,   112,   113,   114,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    -1,    55,    56,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    95,    96,    -1,    -1,    -1,   100,    -1,
      -1,    -1,    -1,    -1,   106,    -1,   108,    -1,   110,    -1,
     112,   113,   114,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    -1,    55,    56,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    95,    96,    -1,
      -1,    -1,   100,    -1,    -1,    -1,    -1,    -1,   106,    -1,
     108,    -1,   110,    -1,   112,   113,   114,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      -1,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    95,    96,    -1,    -1,    -1,   100,    -1,    -1,    -1,
      -1,    -1,   106,    -1,   108,    -1,   110,    -1,   112,   113,
     114,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    -1,    55,    56,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    -1,    -1,    -1,    87,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    95,    96,    -1,    -1,    -1,
     100,    -1,    -1,    -1,    -1,    -1,   106,    -1,   108,    -1,
     110,    -1,   112,   113,   114,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    -1,    55,
      56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,
      -1,    87,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    95,
      96,    -1,    -1,    -1,   100,    -1,    -1,    -1,    -1,    -1,
     106,    -1,   108,    -1,   110,    -1,   112,   113,   114,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    -1,    55,    56,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    95,    96,    -1,    -1,    -1,   100,    -1,
      -1,    -1,    -1,    -1,   106,    -1,   108,    -1,   110,    -1,
     112,   113,   114,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    -1,    55,    56,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    87,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    95,    96,    -1,
      -1,    -1,   100,    -1,    -1,    -1,    -1,    -1,   106,    -1,
     108,    -1,   110,    -1,   112,   113,   114,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      -1,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    95,    96,    -1,    -1,    -1,   100,    -1,    -1,    -1,
      -1,    -1,   106,    -1,   108,    -1,   110,    -1,   112,   113,
     114,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    -1,    55,    56,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    -1,    -1,    -1,    87,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    95,    96,    -1,    -1,    -1,
     100,    -1,    -1,    -1,    -1,    -1,   106,    -1,   108,    -1,
     110,    -1,   112,   113,   114,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    -1,    55,
      56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,
      -1,    87,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    95,
      96,    -1,    -1,    -1,   100,    -1,    -1,    -1,    -1,    -1,
     106,    -1,   108,    -1,   110,    -1,   112,   113,   114,     1,
      -1,     3,     4,     5,     6,     7,     8,    -1,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    -1,    55,    56,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,     3,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    -1,    -1,    -1,    87,    -1,    -1,    -1,    -1,
      -1,    26,    27,    95,    96,    -1,    -1,    -1,   100,    -1,
      -1,    -1,    37,    -1,   106,    -1,   108,    -1,   110,    -1,
     112,   113,   114,    48,    -1,    50,    -1,    52,    -1,    -1,
      55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      95,    96,    -1,    -1,    -1,   100,    -1,    -1,    -1,    37,
      -1,    -1,    -1,   108,    -1,   110,    -1,   112,   113,   114,
      48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    95,    96,    -1,
      -1,    -1,   100,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     108,    -1,   110,    -1,   112,   113,   114,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    95,    96,    -1,    -1,    -1,   100,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   108,    -1,   110,
      -1,   112,   113,   114,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    95,    96,    -1,    -1,    -1,   100,    -1,    -1,    -1,
      37,    -1,    -1,    -1,   108,    -1,   110,    -1,   112,   113,
     114,    48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,
      -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    95,    96,
      -1,    -1,    -1,   100,    -1,    -1,    -1,    37,    -1,    -1,
      -1,   108,    -1,   110,    -1,   112,   113,   114,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    95,    96,    -1,    -1,    -1,
     100,    -1,    -1,    -1,    37,    -1,    -1,    -1,   108,    -1,
     110,    -1,   112,   113,   114,    48,    -1,    50,    -1,    52,
      -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    95,    96,    -1,    -1,    -1,   100,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   108,    -1,   110,    -1,   112,
     113,   114,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    95,
      96,    -1,    -1,    -1,   100,    -1,    -1,    -1,    37,    -1,
      -1,    -1,   108,    -1,   110,    -1,   112,   113,   114,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    95,    96,    -1,    -1,
      -1,   100,    -1,    -1,    -1,    37,    -1,    -1,    -1,   108,
      -1,   110,    -1,   112,   113,   114,    48,    -1,    50,    -1,
      52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    95,    96,    -1,    -1,    -1,   100,    -1,
      -1,    -1,    37,    -1,    -1,    -1,   108,    -1,   110,    -1,
     112,   113,   114,    48,    -1,    50,    -1,    52,    -1,    -1,
      55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      95,    96,    -1,    -1,    -1,   100,    -1,    -1,    -1,    37,
      -1,    -1,    -1,   108,    -1,   110,    -1,   112,   113,   114,
      48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    95,    96,    -1,
      -1,    -1,   100,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     108,    -1,   110,    -1,   112,   113,   114,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    95,    96,    -1,    -1,    -1,   100,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   108,    -1,   110,
      -1,   112,   113,   114,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    95,    96,    -1,    -1,    59,   100,    61,    62,    -1,
      37,    -1,    -1,    -1,   108,    -1,   110,    -1,   112,   113,
     114,    48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,
       1,    -1,    59,    -1,    61,    -1,    90,    91,    92,    93,
      94,    -1,    -1,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,    -1,    -1,    83,    -1,   112,   113,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    95,    96,
      -1,    -1,    -1,   100,    -1,    -1,    47,    -1,    -1,    -1,
      -1,   108,    -1,   110,    -1,   112,   113,   114,    59,    -1,
      61,    62,    -1,     3,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    -1,    -1,    -1,    80,
      81,    82,    -1,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    -1,    -1,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,    47,    -1,    -1,
      -1,   112,   113,    -1,    -1,    -1,    -1,    -1,    -1,    59,
      -1,    61,    62,    -1,     3,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    -1,    -1,    -1,
      80,    81,    82,    -1,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    94,    -1,    -1,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,   107,    47,    -1,
      -1,    -1,   112,   113,    -1,    -1,    -1,    -1,    -1,     3,
      59,    -1,    61,    62,    -1,    -1,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    -1,    -1,
      -1,    80,    81,    82,    -1,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    -1,    -1,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,    -1,
      -1,    -1,    -1,   112,   113,    59,    -1,    61,    62,    -1,
       3,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    -1,    -1,    79,    80,    81,    82,    -1,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    -1,    -1,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,    47,    -1,    -1,    -1,   112,   113,
      -1,    -1,    -1,    -1,    -1,     3,    59,    -1,    61,    62,
      -1,    -1,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    -1,    -1,    -1,    80,    81,    82,
      -1,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    -1,    -1,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,    -1,    -1,    -1,    -1,   112,
     113,    59,     3,    61,    62,    -1,    -1,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    -1,
      -1,    -1,    80,    81,    82,    -1,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    -1,    -1,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
      -1,     3,    -1,    -1,   112,   113,    -1,    -1,    59,    -1,
      61,    62,    -1,    -1,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    -1,    -1,    -1,    80,
      81,    82,    -1,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    -1,    -1,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,    59,     3,    61,
      62,   112,   113,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    -1,    80,    81,
      82,    -1,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    -1,    -1,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,   107,    -1,     3,    -1,    -1,
     112,   113,    -1,    -1,    59,    -1,    61,    62,    -1,    -1,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    -1,    -1,    -1,    80,    81,    82,    -1,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      -1,    -1,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,    59,     3,    61,    62,   112,   113,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    -1,    -1,    -1,    80,    81,    82,    -1,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    -1,
      -1,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,    -1,     3,    -1,    -1,   112,   113,    -1,    -1,
      59,    -1,    61,    62,    -1,    -1,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    -1,    -1,
      -1,    80,    81,    82,    -1,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    -1,    -1,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,    59,
       3,    61,    62,   112,   113,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    -1,    -1,    -1,
      80,    81,    82,    -1,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    94,    -1,    -1,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,   107,    -1,     3,
      -1,    -1,   112,   113,    -1,    -1,    59,    -1,    61,    62,
      -1,    -1,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    -1,    -1,    -1,    80,    81,    82,
      -1,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    -1,    -1,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,    59,     3,    61,    62,   112,
     113,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    -1,    -1,    -1,    80,    81,    82,    -1,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    -1,    -1,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,    -1,    -1,    -1,    -1,   112,   113,
      -1,    -1,    59,    -1,    61,    62,    -1,    -1,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      -1,    -1,    -1,    80,    81,    82,    -1,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    -1,    -1,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,    -1,    -1,    -1,    -1,   112,   113,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      37,    -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,
      47,    48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,
      -1,    -1,    59,    60,    61,    -1,    63,    26,    27,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    83,    -1,    47,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    95,    96,
      59,    -1,    61,   100,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   108,    -1,   110,    -1,   112,   113,   114,    -1,    -1,
      -1,    -1,    -1,    -1,    83,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    95,    96,    -1,    -1,
      -1,   100,    -1,    -1,    -1,   104,    -1,    26,    27,   108,
      -1,   110,    -1,   112,   113,   114,    -1,    -1,    37,    -1,
      -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    60,    61,    -1,    -1,    26,    27,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    83,    -1,    -1,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    95,    96,    59,    60,
      61,   100,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   108,
      -1,   110,    -1,   112,   113,   114,    -1,    -1,    -1,    -1,
      -1,    -1,    83,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    95,    96,    -1,    -1,    -1,   100,
      -1,    -1,    -1,    -1,    -1,    26,    27,   108,    -1,   110,
      -1,   112,   113,   114,    -1,    -1,    37,    -1,    -1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    58,    59,    -1,
      61,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    83,    -1,    -1,    48,    -1,    50,    -1,    52,
      -1,    -1,    55,    56,    95,    96,    59,    60,    61,   100,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   108,    -1,   110,
      -1,   112,   113,   114,    -1,    -1,    -1,    -1,    -1,    -1,
      83,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    95,    96,    -1,    -1,    -1,   100,    -1,    -1,
      -1,    -1,    -1,    26,    27,   108,    -1,   110,    -1,   112,
     113,   114,    -1,    -1,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    48,    -1,    50,    -1,    52,
      -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      83,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    95,    96,    -1,    -1,    -1,   100,    -1,    -1,
      47,    -1,    -1,    -1,    -1,   108,    -1,   110,    -1,   112,
     113,   114,    59,    -1,    61,    62,    63,    -1,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      -1,    -1,    -1,    80,    81,    82,    -1,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    -1,    47,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,    59,    60,    61,    62,   112,   113,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    -1,
      -1,    -1,    80,    81,    82,    -1,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    -1,    47,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
      59,    60,    61,    62,   112,   113,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    -1,    -1,
      -1,    80,    81,    82,    -1,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    -1,    -1,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,    -1,
      -1,    -1,    -1,   112,   113,    58,    59,    -1,    61,    62,
      -1,    -1,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    -1,    -1,    -1,    80,    81,    82,
      -1,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    -1,    -1,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,    59,    60,    61,    62,   112,
     113,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    -1,    -1,    -1,    80,    81,    82,    -1,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    -1,    -1,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,    -1,    -1,    -1,    -1,   112,   113,
      59,    -1,    61,    62,    63,    -1,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    -1,    -1,
      -1,    80,    81,    82,    -1,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    -1,    -1,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,    59,
      60,    61,    62,   112,   113,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    -1,    -1,    -1,
      80,    81,    82,    -1,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    94,    -1,    -1,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,   107,    59,    60,
      61,    62,   112,   113,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    -1,    -1,    -1,    80,
      81,    82,    -1,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    94,    -1,    -1,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,   107,    59,    -1,    61,
      62,   112,   113,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    -1,    80,    81,
      82,    -1,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    -1,    -1,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,   107,    59,    -1,    61,    62,
     112,   113,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    76,    -1,    -1,    79,    80,    81,    82,
      -1,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    -1,    -1,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,    59,    -1,    61,    62,   112,
     113,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    80,    81,    82,    -1,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    -1,    -1,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,    59,    -1,    61,    62,   112,   113,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    82,    -1,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      -1,    -1,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,    59,    -1,    61,    62,   112,   113,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    -1,
      -1,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,    59,    -1,    61,    62,   112,   113,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    -1,    -1,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
     107,    -1,    -1,    -1,    -1,   112,   113
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   116,   117,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    45,    46,    48,
      50,    51,    52,    53,    54,    55,    56,    57,    59,    61,
      83,    87,    95,    96,   100,   106,   108,   110,   112,   113,
     114,   118,   119,   121,   122,   123,   125,   126,   128,   129,
     130,   132,   133,   141,   142,   143,   148,   149,   150,   157,
     159,   172,   174,   183,   184,   186,   193,   194,   195,   197,
     199,   207,   208,   209,   210,   212,   214,   217,   218,   219,
     222,   240,   245,   250,   254,   255,   256,   257,   258,   259,
     260,   262,   265,   267,   270,   271,   272,   273,   274,     3,
      64,     3,     1,     6,   124,   256,     1,    48,   258,     1,
       3,     1,     3,    14,     1,   258,     1,   256,   277,     1,
     258,     1,     1,   258,     1,   258,   274,     1,     3,    47,
       1,   258,     1,     6,     1,     6,     1,     3,   258,   251,
     268,     1,     6,     7,     1,   258,   260,     1,     6,     1,
       3,    47,     1,   258,     1,     3,     6,   211,     1,     6,
      33,   213,     1,     6,   215,   216,     1,     6,   263,   266,
       1,   258,    60,   258,   275,    47,     6,   258,    47,    60,
      63,   258,   274,   278,   258,     1,     3,   274,   258,   258,
     258,     1,     3,   274,   258,     6,    27,   258,   258,     4,
     256,   127,    32,    34,    48,   122,   131,   122,   158,   173,
     185,    49,   202,   205,   206,   122,     1,    59,     1,     6,
     220,   221,   220,     3,    59,    61,    62,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    80,
      81,    82,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   112,   113,   259,    76,    79,   258,
       3,     3,    79,    76,     3,    47,     3,    47,     3,     3,
       3,     3,    47,     3,    47,     3,    79,    92,     3,     3,
       3,     3,     3,     3,     1,    78,     3,   122,     3,     3,
       3,   223,     3,   246,     3,     3,     6,   252,   253,     1,
       6,   200,   201,   269,     3,     3,     3,     3,     3,     3,
      76,     3,    47,     3,     3,    92,     3,     3,    79,     3,
       1,     6,     7,     1,     3,    33,    79,     3,    76,     3,
      79,     3,     1,    59,   264,   264,     3,     3,     1,    60,
      79,   276,   241,    58,    60,   258,    60,    47,    63,     1,
      60,     1,    60,    79,     3,     3,     3,     3,   140,   140,
     160,   175,   140,     1,     3,    47,   140,   203,   204,     3,
       1,   200,     3,     3,    79,     9,   220,     3,    58,   274,
     104,   258,     6,   258,   258,   258,   258,   258,   258,   258,
     258,   258,   258,   258,   258,     1,   258,   258,   258,   258,
     258,   258,   258,   258,   258,     6,   258,   258,   258,   258,
     258,   258,   258,   258,   258,   258,   258,   258,   258,   258,
     258,   258,   258,   256,   258,   256,     1,   258,     3,   274,
       1,    59,   224,   225,     1,    33,   227,   247,     3,    79,
       3,    79,    63,     1,   255,     1,   258,     6,     3,     3,
      76,     3,    76,     3,     6,     7,     6,     4,     6,     7,
     100,   120,   216,     3,   200,   202,   202,   258,     3,    60,
      60,   258,   258,    60,   258,     9,   122,    16,    17,   134,
     136,   137,   139,     1,     3,    23,    24,   161,   166,   168,
       1,     3,    23,   166,   176,    30,   187,   188,   189,   190,
       3,    47,     9,   140,   122,   198,     1,    58,     6,     3,
       3,     1,    58,   258,    60,    79,     1,    47,     3,    79,
      76,     3,     3,    47,     3,     3,   200,   233,   227,     3,
       6,   228,   229,     3,   248,     1,   253,   201,   258,     3,
       3,     3,     3,     6,     6,     3,    77,    78,     3,    77,
      78,     4,     1,    58,   140,   140,   242,    47,    60,    63,
       3,     1,     3,     1,   258,     9,   135,   138,     3,     1,
       6,     7,     8,   120,   170,   171,     1,     9,   167,     3,
       1,     4,     6,   181,   182,     9,     1,     3,     4,     6,
      92,   191,   192,     9,   189,   140,     3,     9,    58,   196,
       3,    47,   261,    60,   274,     1,   258,   274,   258,   144,
     145,     1,    58,     3,     6,    38,    48,    49,    94,   194,
     234,   235,   237,   238,     3,    59,   230,    79,     3,   194,
     235,   237,   238,   249,     3,     3,     6,     6,     6,     6,
       3,     9,     9,     3,     6,     9,   243,   258,   258,     3,
       3,     3,     3,   140,   140,     3,    47,    78,     3,    47,
      79,     3,     3,    47,   169,     3,    47,     3,    47,    79,
       3,     3,   256,     3,    79,    92,     3,     3,    47,    58,
      58,     3,    19,    20,    21,   122,   146,   147,   151,   153,
     155,   122,   226,    76,     3,     6,     1,     6,    83,   239,
       9,     6,    27,   231,   232,   255,   229,     9,     3,     3,
       3,     3,     3,    76,    79,   244,     3,    60,   134,   164,
     165,   120,   162,   163,   171,   140,   122,   179,   180,   177,
     178,   182,     3,   192,   256,     3,     1,     3,    47,     1,
       3,    47,     1,     3,    47,     9,   146,    58,   258,   236,
      76,     3,     6,     3,    79,     3,    58,    79,     3,   255,
     140,   122,   140,   122,   140,   122,   140,   122,     3,     3,
     152,   122,     3,   154,   122,     3,   156,   122,     3,     3,
     202,   258,     6,    83,   232,   244,   140,   140,   140,   140,
       3,     6,     9,     9,     9,     9,     3,     3,     3,     3
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */






/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  /* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;

  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:
#line 210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); }
    break;

  case 7:
#line 211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); }
    break;

  case 8:
#line 215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat)=0; }
    break;

  case 10:
#line 218 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 11:
#line 223 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 12:
#line 228 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 13:
#line 233 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 14:
#line 238 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 20:
#line 250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); }
    break;

  case 21:
#line 255 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), false );
      }
    break;

  case 22:
#line 261 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), true );
      }
    break;

  case 23:
#line 267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
      }
    break;

  case 24:
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); }
    break;

  case 25:
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; }
    break;

  case 26:
#line 275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; }
    break;

  case 27:
#line 276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; }
    break;

  case 28:
#line 277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; }
    break;

  case 29:
#line 278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;}
    break;

  case 30:
#line 283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); }
    break;

  case 31:
#line 285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      }
    break;

  case 32:
#line 291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (6)].fal_adecl)->size() != (yyvsp[(5) - (6)].fal_adecl)->size() + 1 )
         {
            COMPILER->raiseError(Falcon::e_unpack_size );
         }
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (6)].fal_adecl) );

         COMPILER->defineVal( first );
         (yyvsp[(5) - (6)].fal_adecl)->pushFront( (yyvsp[(3) - (6)].fal_val) );
         Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, second ) ) );
      }
    break;

  case 51:
#line 326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   }
    break;

  case 52:
#line 332 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   }
    break;

  case 53:
#line 341 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; }
    break;

  case 54:
#line 343 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); }
    break;

  case 55:
#line 347 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 56:
#line 354 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 57:
#line 361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 58:
#line 369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 59:
#line 370 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 60:
#line 371 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; }
    break;

  case 61:
#line 375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 62:
#line 376 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 63:
#line 377 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 64:
#line 381 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
    break;

  case 65:
#line 389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 66:
#line 396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      }
    break;

  case 67:
#line 406 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 68:
#line 407 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; }
    break;

  case 69:
#line 411 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 70:
#line 412 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 73:
#line 419 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
    break;

  case 76:
#line 429 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); }
    break;

  case 77:
#line 434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
    break;

  case 79:
#line 446 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 80:
#line 447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; }
    break;

  case 82:
#line 452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   }
    break;

  case 83:
#line 459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      }
    break;

  case 84:
#line 468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      }
    break;

  case 85:
#line 476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      }
    break;

  case 86:
#line 486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      }
    break;

  case 87:
#line 495 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      }
    break;

  case 88:
#line 504 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (5)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (5)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( LINE, new Falcon::Value(decl), (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 89:
#line 520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 90:
#line 528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
          Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (5)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( CURRENT_LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (5)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( CURRENT_LINE, new Falcon::Value(decl), (yyvsp[(4) - (5)].fal_val) );

         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 91:
#line 544 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(7) - (7)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(7) - (7)].fal_stat) );
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 92:
#line 554 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 93:
#line 559 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 96:
#line 571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      }
    break;

  case 100:
#line 585 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getLoop());
         if ( f == 0 || f->type() != Falcon::Statement::t_forin )
         {
            COMPILER->raiseError( Falcon::e_syn_fordot );
            delete (yyvsp[(2) - (3)].fal_val);
            (yyval.fal_stat) = 0;
         }
         else {
            (yyval.fal_stat) = new Falcon::StmtFordot( LINE, (yyvsp[(2) - (3)].fal_val) );
         }
      }
    break;

  case 101:
#line 598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      }
    break;

  case 102:
#line 606 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 103:
#line 610 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 104:
#line 616 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 105:
#line 622 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      }
    break;

  case 106:
#line 629 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 107:
#line 634 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 108:
#line 643 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   }
    break;

  case 109:
#line 652 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      }
    break;

  case 110:
#line 664 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 111:
#line 666 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 112:
#line 675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
    break;

  case 113:
#line 679 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      }
    break;

  case 114:
#line 691 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 115:
#line 692 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 116:
#line 701 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); }
    break;

  case 117:
#line 705 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 // Apparently you get a segfault without it.
		 // (Note that the formiddle: version below does *not* need it
		 f->middleBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->middleBlock() );
      }
    break;

  case 118:
#line 719 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 119:
#line 721 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->middleBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 120:
#line 730 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
    break;

  case 121:
#line 734 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 122:
#line 742 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 123:
#line 751 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 124:
#line 753 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 127:
#line 762 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); }
    break;

  case 129:
#line 768 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 131:
#line 778 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 132:
#line 786 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 133:
#line 790 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 135:
#line 802 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 136:
#line 812 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 138:
#line 821 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      }
    break;

  case 142:
#line 835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); }
    break;

  case 144:
#line 839 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 147:
#line 851 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }
    break;

  case 148:
#line 860 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 149:
#line 872 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].stringp) );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 150:
#line 883 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (yyvsp[(1) - (3)].integer) ), new Falcon::Value( (yyvsp[(3) - (3)].integer) ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 151:
#line 894 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 152:
#line 914 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 153:
#line 922 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 154:
#line 931 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 155:
#line 933 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 158:
#line 942 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); }
    break;

  case 160:
#line 948 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 162:
#line 958 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 163:
#line 967 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 164:
#line 971 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 166:
#line 983 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 167:
#line 993 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 171:
#line 1007 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 172:
#line 1019 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 173:
#line 1040 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_adecl), (yyvsp[(2) - (5)].fal_adecl) );
      }
    break;

  case 174:
#line 1044 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      }
    break;

  case 175:
#line 1048 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; }
    break;

  case 176:
#line 1056 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   }
    break;

  case 177:
#line 1063 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }
    break;

  case 178:
#line 1073 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
    break;

  case 180:
#line 1082 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); }
    break;

  case 186:
#line 1102 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }
         // but continue by pushing this new context
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }
    break;

  case 187:
#line 1120 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }

         // but continue by pushing this new context
         COMPILER->defineVal( (yyvsp[(3) - (4)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(3) - (4)].fal_val) );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      }
    break;

  case 188:
#line 1140 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 189:
#line 1150 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 190:
#line 1161 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }
    break;

  case 193:
#line 1174 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 194:
#line 1186 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 195:
#line 1208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 196:
#line 1209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; }
    break;

  case 197:
#line 1221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 198:
#line 1227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 200:
#line 1236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 201:
#line 1237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 202:
#line 1240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
    break;

  case 204:
#line 1245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 205:
#line 1246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 206:
#line 1253 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String *func_name;
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::String complete_name = stmt_cls->symbol()->name() + "." + *(yyvsp[(2) - (2)].stringp);
            func_name = COMPILER->addString( complete_name );
         }
         else
            func_name = (yyvsp[(2) - (2)].stringp);

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( func_name );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( func_name );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         // and eventually add it as a class property
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
            if ( cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) ) {
               COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (2)].stringp) );
            }
            else {
                cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef( sym ) );
            }
         }

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 210:
#line 1314 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), (yyvsp[(1) - (1)].stringp) );
            COMPILER->module()->addSymbol( sym );
            func->addParameter( sym );
         }
      }
    break;

  case 212:
#line 1331 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 213:
#line 1337 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 214:
#line 1342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 215:
#line 1348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 217:
#line 1357 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); }
    break;

  case 219:
#line 1362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
    break;

  case 220:
#line 1372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 221:
#line 1375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; }
    break;

  case 222:
#line 1384 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 223:
#line 1391 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         if ( COMPILER->getFunction() == 0 )
         {
            COMPILER->raiseError(Falcon::e_pass_outside );
            /*delete $2;
            delete $4;*/
            (yyval.fal_stat) = 0;
         }
         else {
            COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         }
      }
    break;

  case 224:
#line 1406 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      }
    break;

  case 225:
#line 1412 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      }
    break;

  case 226:
#line 1424 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = (yyvsp[(4) - (5)].fal_val); //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *(yyvsp[(2) - (5)].stringp), val, LINE );
         // we don't need the expression anymore
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 227:
#line 1434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      }
    break;

  case 228:
#line 1439 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      }
    break;

  case 229:
#line 1451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 230:
#line 1460 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      }
    break;

  case 231:
#line 1467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      }
    break;

  case 232:
#line 1475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      }
    break;

  case 233:
#line 1480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      }
    break;

  case 234:
#line 1488 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (3)].fal_genericList) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 235:
#line 1493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 236:
#line 1498 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 237:
#line 1503 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( symName, (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );

         (yyval.fal_stat) = 0;
      }
    break;

  case 238:
#line 1523 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( symName, (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 239:
#line 1542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 240:
#line 1547 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 241:
#line 1552 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 242:
#line 1557 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (4)].fal_genericList)->begin();
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            delete symName;
            li = li->next();
         }
         delete (yyvsp[(2) - (4)].fal_genericList);

         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 243:
#line 1571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 244:
#line 1576 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 245:
#line 1581 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 246:
#line 1586 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 247:
#line 1591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 248:
#line 1599 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *(yyvsp[(1) - (1)].stringp) ) );
         (yyval.fal_genericList) = lst;
      }
    break;

  case 249:
#line 1605 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(1) - (3)].fal_genericList)->pushBack( new Falcon::String( *(yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_genericList) = (yyvsp[(1) - (3)].fal_genericList);
      }
    break;

  case 250:
#line 1617 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 251:
#line 1622 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 254:
#line 1635 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 255:
#line 1639 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 256:
#line 1643 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 257:
#line 1657 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 258:
#line 1664 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 260:
#line 1672 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); }
    break;

  case 262:
#line 1676 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); }
    break;

  case 264:
#line 1682 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         }
    break;

  case 265:
#line 1686 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         }
    break;

  case 268:
#line 1695 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   }
    break;

  case 269:
#line 1706 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );
      }
    break;

  case 270:
#line 1740 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 272:
#line 1768 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 275:
#line 1776 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 276:
#line 1777 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 281:
#line 1794 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol((yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            if ( (yyvsp[(2) - (2)].fal_adecl) != 0 )
            {
               // save the carried
               Falcon::ListElement *iter = (yyvsp[(2) - (2)].fal_adecl)->begin();
               while( iter != 0 )
               {
                  Falcon::Value *val = (Falcon::Value *) iter->data();
                  idef->addParameter( val->genVarDefSym() );
                  iter = iter->next();
               }

               // dispose of the carrier
               delete (yyvsp[(2) - (2)].fal_adecl);
            }
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
         }
      }
    break;

  case 282:
#line 1827 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; }
    break;

  case 283:
#line 1832 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
   }
    break;

  case 284:
#line 1838 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 285:
#line 1839 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 287:
#line 1845 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      }
    break;

  case 288:
#line 1853 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 292:
#line 1863 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 293:
#line 1866 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   }
    break;

  case 295:
#line 1888 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         if( cls->initGiven() ) {
            COMPILER->raiseError(Falcon::e_init_given );
         }
         else
         {
            cls->initGiven( true );
            Falcon::StmtFunction *func = cls->ctorFunction();
            if ( func == 0 ) {
               func = COMPILER->buildCtorFor( cls );
            }

            // prepare the statement allocation context
            COMPILER->pushContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      }
    break;

  case 296:
#line 1912 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 297:
#line 1921 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(4) - (5)].fal_val)->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *(yyvsp[(2) - (5)].stringp);
         Falcon::Symbol *sym = COMPILER->addGlobalVar( COMPILER->addString(prop_name), def );
         if( clsdef->hasProperty( *(yyvsp[(2) - (5)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (5)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(2) - (5)].stringp), new Falcon::VarDef( Falcon::VarDef::t_reference, sym) );
      }
      else {
         COMPILER->raiseError(Falcon::e_static_const );
      }
      delete (yyvsp[(4) - (5)].fal_val); // the expression is not needed anymore
      (yyval.fal_stat) = 0; // we don't add any statement
   }
    break;

  case 298:
#line 1943 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(3) - (4)].fal_val)->genVarDef();

      if ( def != 0 ) {
         if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), def );
         delete (yyvsp[(3) - (4)].fal_val); // the expression is not needed anymore
         (yyval.fal_stat) = 0; // we don't add any statement
      }
      else {
         // create anyhow a nil property
          if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), new Falcon::VarDef() );
         // but also prepare a statement to be executed by the auto-constructor.
         (yyval.fal_stat) = new Falcon::StmtVarDef( LINE, (yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
   }
    break;

  case 301:
#line 1973 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   }
    break;

  case 302:
#line 1980 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      }
    break;

  case 303:
#line 1988 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      }
    break;

  case 304:
#line 1994 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      }
    break;

  case 305:
#line 2000 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      }
    break;

  case 306:
#line 2013 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
            sym->setEnum( true );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );

         COMPILER->resetEnum();
      }
    break;

  case 307:
#line 2047 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 311:
#line 2064 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 312:
#line 2069 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 315:
#line 2084 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *(yyvsp[(2) - (2)].stringp);
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( COMPILER->addString( cl_name ) );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         cls->singleton( sym );

         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      }
    break;

  case 316:
#line 2126 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( !cd->inheritance().empty() )
            {
               COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 318:
#line 2151 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 322:
#line 2163 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 323:
#line 2166 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   }
    break;

  case 325:
#line 2194 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 326:
#line 2199 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
      }
    break;

  case 329:
#line 2214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 330:
#line 2221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 331:
#line 2236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 332:
#line 2237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 333:
#line 2238 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 334:
#line 2248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 335:
#line 2249 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 336:
#line 2250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 337:
#line 2251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 338:
#line 2252 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 339:
#line 2253 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 340:
#line 2258 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            val = new Falcon::Value();
            val->setSymdef( (yyvsp[(1) - (1)].stringp) );
            // warning: the symbol is still undefined.
            COMPILER->addSymdef( val );
         }
         else {
            val = new Falcon::Value( sym );
         }
         (yyval.fal_val) = val;
     }
    break;

  case 342:
#line 2276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 343:
#line 2277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); }
    break;

  case 346:
#line 2290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( (yyvsp[(2) - (2)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 347:
#line 2291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString("self") ); /* do not add the symbol to the compiler */ }
    break;

  case 348:
#line 2292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 349:
#line 2293 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_fbind, new Falcon::Value((yyvsp[(1) - (3)].stringp)), (yyvsp[(3) - (3)].fal_val)) ); }
    break;

  case 350:
#line 2294 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 351:
#line 2295 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 352:
#line 2296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 353:
#line 2297 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 354:
#line 2298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 355:
#line 2299 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 356:
#line 2300 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 357:
#line 2301 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 358:
#line 2302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 359:
#line 2303 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 360:
#line 2304 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 361:
#line 2305 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 362:
#line 2306 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 363:
#line 2307 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 364:
#line 2308 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 365:
#line 2309 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 366:
#line 2310 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 367:
#line 2311 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 368:
#line 2312 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 369:
#line 2313 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 370:
#line 2314 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 371:
#line 2315 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 372:
#line 2316 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 373:
#line 2317 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 374:
#line 2318 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 375:
#line 2319 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 376:
#line 2320 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 377:
#line 2321 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 378:
#line 2322 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 379:
#line 2323 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 380:
#line 2324 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 381:
#line 2325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 382:
#line 2326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 383:
#line 2327 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 390:
#line 2335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 391:
#line 2340 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 392:
#line 2344 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 393:
#line 2349 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 394:
#line 2355 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 397:
#line 2367 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 398:
#line 2372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 399:
#line 2379 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 400:
#line 2380 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 401:
#line 2381 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 402:
#line 2382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 403:
#line 2383 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 404:
#line 2384 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 405:
#line 2385 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 406:
#line 2386 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 407:
#line 2387 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 408:
#line 2388 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 409:
#line 2389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 410:
#line 2390 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 411:
#line 2395 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 412:
#line 2398 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 413:
#line 2401 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 414:
#line 2404 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 415:
#line 2407 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 416:
#line 2414 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 417:
#line 2420 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 418:
#line 2424 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 419:
#line 2425 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 420:
#line 2434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 421:
#line 2468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 423:
#line 2476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 424:
#line 2480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 425:
#line 2487 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 426:
#line 2520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 427:
#line 2532 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      }
    break;

  case 428:
#line 2564 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            COMPILER->checkLocalUndefined();
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 430:
#line 2575 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      }
    break;

  case 431:
#line 2584 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 432:
#line 2589 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 433:
#line 2596 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 434:
#line 2603 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 435:
#line 2612 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 436:
#line 2614 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 437:
#line 2618 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 438:
#line 2625 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 439:
#line 2627 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 440:
#line 2631 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 441:
#line 2639 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 442:
#line 2640 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 443:
#line 2642 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 444:
#line 2649 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 445:
#line 2650 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 446:
#line 2654 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 447:
#line 2655 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 450:
#line 2662 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 451:
#line 2668 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 452:
#line 2675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 453:
#line 2676 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;


/* Line 1267 of yacc.c.  */
#line 6451 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 2680 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


