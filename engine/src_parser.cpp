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
     ASSIGN_POW = 319,
     ASSIGN_SHL = 320,
     ASSIGN_SHR = 321,
     ASSIGN_BXOR = 322,
     ASSIGN_BOR = 323,
     ASSIGN_BAND = 324,
     ASSIGN_MOD = 325,
     ASSIGN_DIV = 326,
     ASSIGN_MUL = 327,
     ASSIGN_SUB = 328,
     ASSIGN_ADD = 329,
     OP_EQ = 330,
     OP_TO = 331,
     COMMA = 332,
     QUESTION = 333,
     OR = 334,
     AND = 335,
     NOT = 336,
     LE = 337,
     GE = 338,
     LT = 339,
     GT = 340,
     NEQ = 341,
     EEQ = 342,
     PROVIDES = 343,
     OP_NOTIN = 344,
     OP_IN = 345,
     HASNT = 346,
     HAS = 347,
     DIESIS = 348,
     ATSIGN = 349,
     CAP_CAP = 350,
     VBAR_VBAR = 351,
     AMPER_AMPER = 352,
     MINUS = 353,
     PLUS = 354,
     PERCENT = 355,
     SLASH = 356,
     STAR = 357,
     POW = 358,
     SHR = 359,
     SHL = 360,
     TILDE = 361,
     NEG = 362,
     AMPER = 363,
     BANG = 364,
     DECREMENT = 365,
     INCREMENT = 366,
     DOLLAR = 367
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
#define ASSIGN_POW 319
#define ASSIGN_SHL 320
#define ASSIGN_SHR 321
#define ASSIGN_BXOR 322
#define ASSIGN_BOR 323
#define ASSIGN_BAND 324
#define ASSIGN_MOD 325
#define ASSIGN_DIV 326
#define ASSIGN_MUL 327
#define ASSIGN_SUB 328
#define ASSIGN_ADD 329
#define OP_EQ 330
#define OP_TO 331
#define COMMA 332
#define QUESTION 333
#define OR 334
#define AND 335
#define NOT 336
#define LE 337
#define GE 338
#define LT 339
#define GT 340
#define NEQ 341
#define EEQ 342
#define PROVIDES 343
#define OP_NOTIN 344
#define OP_IN 345
#define HASNT 346
#define HAS 347
#define DIESIS 348
#define ATSIGN 349
#define CAP_CAP 350
#define VBAR_VBAR 351
#define AMPER_AMPER 352
#define MINUS 353
#define PLUS 354
#define PERCENT 355
#define SLASH 356
#define STAR 357
#define POW 358
#define SHR 359
#define SHL 360
#define TILDE 361
#define NEG 362
#define AMPER 363
#define BANG 364
#define DECREMENT 365
#define INCREMENT 366
#define DOLLAR 367




/* Copy the first part of user declarations.  */
#line 17 "/home/user/Progetti/falcon/core/engine/src_parser.yy"


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
#line 61 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
/* Line 187 of yacc.c.  */
#line 384 "/home/user/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 397 "/home/user/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   6227

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  113
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  163
/* YYNRULES -- Number of rules.  */
#define YYNRULES  436
/* YYNRULES -- Number of states.  */
#define YYNSTATES  791

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   367

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
     105,   106,   107,   108,   109,   110,   111,   112
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      43,    47,    51,    55,    57,    59,    63,    67,    71,    74,
      77,    82,    89,    91,    93,    95,    97,    99,   101,   103,
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
     744,   748,   752,   754,   758,   762,   766,   768,   772,   776,
     780,   782,   786,   790,   794,   798,   803,   807,   810,   814,
     817,   821,   822,   824,   828,   831,   835,   838,   839,   848,
     852,   855,   856,   860,   861,   867,   868,   871,   873,   877,
     880,   881,   885,   887,   891,   893,   895,   897,   898,   901,
     903,   905,   907,   909,   910,   918,   924,   929,   930,   934,
     938,   940,   943,   947,   952,   953,   961,   962,   965,   967,
     972,   975,   977,   979,   980,   989,   992,   995,   996,   999,
    1001,  1003,  1005,  1007,  1008,  1013,  1015,  1019,  1023,  1025,
    1028,  1032,  1036,  1038,  1040,  1042,  1044,  1046,  1048,  1050,
    1052,  1054,  1056,  1058,  1060,  1063,  1067,  1071,  1075,  1079,
    1083,  1087,  1091,  1095,  1099,  1103,  1107,  1110,  1114,  1117,
    1120,  1123,  1126,  1130,  1134,  1138,  1142,  1146,  1150,  1154,
    1157,  1161,  1165,  1169,  1173,  1177,  1180,  1183,  1186,  1189,
    1191,  1193,  1195,  1197,  1199,  1201,  1204,  1206,  1211,  1217,
    1221,  1223,  1225,  1229,  1235,  1239,  1243,  1247,  1251,  1255,
    1259,  1263,  1267,  1271,  1275,  1279,  1283,  1287,  1292,  1297,
    1303,  1311,  1316,  1320,  1321,  1328,  1329,  1336,  1341,  1345,
    1348,  1349,  1356,  1357,  1363,  1365,  1368,  1374,  1380,  1385,
    1389,  1392,  1396,  1400,  1403,  1407,  1411,  1415,  1419,  1424,
    1426,  1430,  1432,  1435,  1437,  1441,  1445
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     114,     0,    -1,   115,    -1,    -1,   115,   116,    -1,   117,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   119,    -1,
     212,    -1,   192,    -1,   220,    -1,   243,    -1,   120,    -1,
     207,    -1,   208,    -1,   210,    -1,   215,    -1,     4,    -1,
      98,     4,    -1,    39,     6,     3,    -1,    39,     7,     3,
      -1,    39,     1,     3,    -1,   121,    -1,     3,    -1,    48,
       1,     3,    -1,    34,     1,     3,    -1,    32,     1,     3,
      -1,     1,     3,    -1,   256,     3,    -1,   272,    75,   256,
       3,    -1,   272,    75,   256,    77,   272,     3,    -1,   123,
      -1,   124,    -1,   141,    -1,   155,    -1,   170,    -1,   128,
      -1,   139,    -1,   140,    -1,   181,    -1,   182,    -1,   191,
      -1,   252,    -1,   248,    -1,   205,    -1,   206,    -1,   146,
      -1,   147,    -1,   148,    -1,   238,    -1,   254,    75,   256,
      -1,   122,    77,   254,    75,   256,    -1,    10,   122,     3,
      -1,    10,     1,     3,    -1,    -1,   126,   125,   138,     9,
       3,    -1,   127,   120,    -1,    11,   256,     3,    -1,    53,
      -1,    11,     1,     3,    -1,    11,   256,    47,    -1,    53,
      47,    -1,    11,     1,    47,    -1,    -1,   130,   129,   138,
     132,     9,     3,    -1,   131,   120,    -1,    15,   256,     3,
      -1,    15,     1,     3,    -1,    15,   256,    47,    -1,    15,
       1,    47,    -1,    -1,   135,    -1,    -1,   134,   133,   138,
      -1,    16,     3,    -1,    16,     1,     3,    -1,    -1,   137,
     136,   138,   132,    -1,    17,   256,     3,    -1,    17,     1,
       3,    -1,    -1,   138,   120,    -1,    12,     3,    -1,    12,
       1,     3,    -1,    13,     3,    -1,    13,    14,     3,    -1,
      13,     1,     3,    -1,    -1,    18,   274,    90,   256,     3,
     142,   144,     9,     3,    -1,    -1,    18,   274,    90,   256,
      47,   143,   120,    -1,    18,   274,    90,     1,     3,    -1,
      18,     1,     3,    -1,    -1,   145,   144,    -1,   120,    -1,
     149,    -1,   151,    -1,   153,    -1,    51,   256,     3,    -1,
      51,     1,     3,    -1,   104,   272,     3,    -1,   104,     3,
      -1,    85,   272,     3,    -1,    85,     3,    -1,   104,     1,
       3,    -1,    85,     1,     3,    -1,    57,    -1,    -1,    19,
       3,   150,   138,     9,     3,    -1,    19,    47,   120,    -1,
      19,     1,     3,    -1,    -1,    20,     3,   152,   138,     9,
       3,    -1,    20,    47,   120,    -1,    20,     1,     3,    -1,
      -1,    21,     3,   154,   138,     9,     3,    -1,    21,    47,
     120,    -1,    21,     1,     3,    -1,    -1,   157,   156,   158,
     164,     9,     3,    -1,    22,   256,     3,    -1,    22,     1,
       3,    -1,    -1,   158,   159,    -1,   158,     1,     3,    -1,
       3,    -1,    -1,    23,   168,     3,   160,   138,    -1,    -1,
      23,   168,    47,   161,   120,    -1,    -1,    23,     1,     3,
     162,   138,    -1,    -1,    23,     1,    47,   163,   120,    -1,
      -1,    -1,   166,   165,   167,    -1,    -1,    24,    -1,    24,
       1,    -1,     3,   138,    -1,    47,   120,    -1,   169,    -1,
     168,    77,   169,    -1,     8,    -1,   118,    -1,     7,    -1,
     118,    76,   118,    -1,     6,    -1,    -1,   172,   171,   173,
     164,     9,     3,    -1,    25,   256,     3,    -1,    25,     1,
       3,    -1,    -1,   173,   174,    -1,   173,     1,     3,    -1,
       3,    -1,    -1,    23,   179,     3,   175,   138,    -1,    -1,
      23,   179,    47,   176,   120,    -1,    -1,    23,     1,     3,
     177,   138,    -1,    -1,    23,     1,    47,   178,   120,    -1,
     180,    -1,   179,    77,   180,    -1,    -1,     4,    -1,     6,
      -1,    28,   272,    76,   272,     3,    -1,    28,   272,     1,
       3,    -1,    28,     1,     3,    -1,    29,    47,   120,    -1,
      -1,   184,   183,   138,   185,     9,     3,    -1,    29,     3,
      -1,    29,     1,     3,    -1,    -1,   186,    -1,   187,    -1,
     186,   187,    -1,   188,   138,    -1,    30,     3,    -1,    30,
      90,   254,     3,    -1,    30,   189,     3,    -1,    30,   189,
      90,   254,     3,    -1,    30,     1,     3,    -1,   190,    -1,
     189,    77,   190,    -1,     4,    -1,     6,    -1,    31,   256,
       3,    -1,    31,     1,     3,    -1,   193,   200,   138,     9,
       3,    -1,   195,   120,    -1,   197,    59,   198,    58,     3,
      -1,    -1,   197,    59,   198,     1,   194,    58,     3,    -1,
     197,     1,     3,    -1,   197,    59,   198,    58,    47,    -1,
      -1,   197,    59,     1,   196,    58,    47,    -1,    48,     6,
      -1,    -1,   199,    -1,   198,    77,   199,    -1,     6,    -1,
      -1,    -1,   203,   201,   138,     9,     3,    -1,    -1,   204,
     202,   120,    -1,    49,     3,    -1,    49,     1,     3,    -1,
      49,    47,    -1,    49,     1,    47,    -1,    40,   258,     3,
      -1,    40,     1,     3,    -1,    43,   256,     3,    -1,    43,
     256,    90,   256,     3,    -1,    43,   256,    90,     1,     3,
      -1,    43,     1,     3,    -1,    41,     6,    75,   253,     3,
      -1,    41,     6,    75,     1,     3,    -1,    41,     1,     3,
      -1,    44,     3,    -1,    44,   209,     3,    -1,    44,     1,
       3,    -1,     6,    -1,   209,    77,     6,    -1,    45,   211,
       3,    -1,    45,     1,     3,    -1,     6,    -1,   209,    77,
       6,    -1,    46,   213,     3,    -1,    46,     1,     3,    -1,
     214,    -1,   213,    77,   214,    -1,     6,    75,     6,    -1,
       6,    75,     7,    -1,     6,    75,   118,    -1,   216,   219,
       9,     3,    -1,   217,   218,     3,    -1,    42,     3,    -1,
      42,     1,     3,    -1,    42,    47,    -1,    42,     1,    47,
      -1,    -1,     6,    -1,   218,    77,     6,    -1,   218,     3,
      -1,   219,   218,     3,    -1,     1,     3,    -1,    -1,    32,
       6,   221,   222,   231,   236,     9,     3,    -1,   223,   225,
       3,    -1,     1,     3,    -1,    -1,    59,   198,    58,    -1,
      -1,    59,   198,     1,   224,    58,    -1,    -1,    33,   226,
      -1,   227,    -1,   226,    77,   227,    -1,     6,   228,    -1,
      -1,    59,   229,    58,    -1,   230,    -1,   229,    77,   230,
      -1,   253,    -1,     6,    -1,    27,    -1,    -1,   231,   232,
      -1,     3,    -1,   192,    -1,   235,    -1,   233,    -1,    -1,
      38,     3,   234,   200,   138,     9,     3,    -1,    49,     6,
      75,   256,     3,    -1,     6,    75,   256,     3,    -1,    -1,
      92,   237,     3,    -1,    92,     1,     3,    -1,     6,    -1,
      81,     6,    -1,   237,    77,     6,    -1,   237,    77,    81,
       6,    -1,    -1,    54,     6,   239,     3,   240,     9,     3,
      -1,    -1,   240,   241,    -1,     3,    -1,     6,    75,   253,
     242,    -1,     6,   242,    -1,     3,    -1,    77,    -1,    -1,
      34,     6,   244,   245,   246,   236,     9,     3,    -1,   225,
       3,    -1,     1,     3,    -1,    -1,   246,   247,    -1,     3,
      -1,   192,    -1,   235,    -1,   233,    -1,    -1,    36,   249,
     250,     3,    -1,   251,    -1,   250,    77,   251,    -1,   250,
      77,     1,    -1,     6,    -1,    35,     3,    -1,    35,   256,
       3,    -1,    35,     1,     3,    -1,     8,    -1,    55,    -1,
      56,    -1,     4,    -1,     5,    -1,     7,    -1,     6,    -1,
     254,    -1,    27,    -1,    26,    -1,   253,    -1,   255,    -1,
      98,   256,    -1,   256,    99,   256,    -1,   256,    98,   256,
      -1,   256,   102,   256,    -1,   256,   101,   256,    -1,   256,
     100,   256,    -1,   256,   103,   256,    -1,   256,    97,   256,
      -1,   256,    96,   256,    -1,   256,    95,   256,    -1,   256,
     105,   256,    -1,   256,   104,   256,    -1,   106,   256,    -1,
     256,    86,   256,    -1,   256,   111,    -1,   111,   256,    -1,
     256,   110,    -1,   110,   256,    -1,   256,    87,   256,    -1,
     256,    85,   256,    -1,   256,    84,   256,    -1,   256,    83,
     256,    -1,   256,    82,   256,    -1,   256,    80,   256,    -1,
     256,    79,   256,    -1,    81,   256,    -1,   256,    92,   256,
      -1,   256,    91,   256,    -1,   256,    90,   256,    -1,   256,
      89,   256,    -1,   256,    88,     6,    -1,   112,   254,    -1,
     112,     4,    -1,    94,   256,    -1,    93,   256,    -1,   265,
      -1,   260,    -1,   263,    -1,   258,    -1,   268,    -1,   270,
      -1,   256,   257,    -1,   269,    -1,   256,    61,   256,    60,
      -1,   256,    61,   102,   256,    60,    -1,   256,    62,     6,
      -1,   271,    -1,   257,    -1,   256,    75,   256,    -1,   256,
      75,   256,    77,   272,    -1,   256,    74,   256,    -1,   256,
      73,   256,    -1,   256,    72,   256,    -1,   256,    71,   256,
      -1,   256,    70,   256,    -1,   256,    64,   256,    -1,   256,
      69,   256,    -1,   256,    68,   256,    -1,   256,    67,   256,
      -1,   256,    65,   256,    -1,   256,    66,   256,    -1,    59,
     256,    58,    -1,    61,    47,    60,    -1,    61,   256,    47,
      60,    -1,    61,    47,   256,    60,    -1,    61,   256,    47,
     256,    60,    -1,    61,   256,    47,   256,    47,   256,    60,
      -1,   256,    59,   272,    58,    -1,   256,    59,    58,    -1,
      -1,   256,    59,   272,     1,   259,    58,    -1,    -1,    48,
     261,   262,   200,   138,     9,    -1,    59,   198,    58,     3,
      -1,    59,   198,     1,    -1,     1,     3,    -1,    -1,    50,
     264,   262,   200,   138,     9,    -1,    -1,    37,   266,   267,
      63,   256,    -1,   198,    -1,     1,     3,    -1,   256,    78,
     256,    47,   256,    -1,   256,    78,   256,    47,     1,    -1,
     256,    78,   256,     1,    -1,   256,    78,     1,    -1,    61,
      60,    -1,    61,   272,    60,    -1,    61,   272,     1,    -1,
      52,    60,    -1,    52,   273,    60,    -1,    52,   273,     1,
      -1,    61,    63,    60,    -1,    61,   275,    60,    -1,    61,
     275,     1,    60,    -1,   256,    -1,   272,    77,   256,    -1,
     256,    -1,   273,   256,    -1,   254,    -1,   274,    77,   254,
      -1,   256,    63,   256,    -1,   275,    77,   256,    63,   256,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   196,   196,   199,   201,   205,   206,   207,   211,   216,
     217,   222,   227,   232,   237,   238,   239,   240,   244,   245,
     249,   255,   261,   269,   270,   271,   272,   273,   274,   279,
     281,   287,   301,   302,   303,   304,   305,   306,   307,   308,
     309,   310,   311,   312,   313,   314,   315,   316,   317,   318,
     319,   323,   329,   337,   339,   344,   344,   358,   366,   367,
     368,   372,   373,   374,   378,   378,   393,   403,   404,   408,
     409,   413,   415,   416,   416,   425,   426,   431,   431,   443,
     444,   447,   449,   455,   464,   472,   482,   491,   501,   500,
     525,   524,   550,   555,   562,   564,   568,   575,   576,   577,
     581,   594,   602,   606,   612,   618,   625,   630,   639,   649,
     649,   663,   672,   676,   676,   689,   698,   702,   702,   718,
     727,   731,   731,   748,   749,   756,   758,   759,   763,   765,
     764,   775,   775,   787,   787,   799,   799,   815,   818,   817,
     830,   831,   832,   835,   836,   842,   843,   847,   856,   868,
     879,   890,   911,   911,   928,   929,   936,   938,   939,   943,
     945,   944,   955,   955,   968,   968,   980,   980,   998,   999,
    1002,  1003,  1015,  1036,  1040,  1045,  1053,  1060,  1059,  1078,
    1079,  1082,  1084,  1088,  1089,  1093,  1098,  1116,  1136,  1146,
    1157,  1165,  1166,  1170,  1182,  1205,  1206,  1213,  1223,  1232,
    1233,  1233,  1237,  1241,  1242,  1242,  1249,  1303,  1305,  1306,
    1310,  1325,  1328,  1327,  1339,  1338,  1353,  1354,  1358,  1359,
    1368,  1372,  1380,  1387,  1402,  1408,  1420,  1430,  1435,  1447,
    1456,  1463,  1471,  1476,  1484,  1488,  1496,  1501,  1513,  1518,
    1526,  1527,  1531,  1535,  1539,  1551,  1558,  1568,  1569,  1572,
    1573,  1576,  1578,  1582,  1589,  1590,  1591,  1603,  1602,  1661,
    1664,  1670,  1672,  1673,  1673,  1679,  1681,  1685,  1686,  1690,
    1724,  1726,  1735,  1736,  1740,  1741,  1750,  1753,  1755,  1759,
    1760,  1763,  1781,  1785,  1785,  1819,  1841,  1868,  1870,  1871,
    1878,  1886,  1892,  1898,  1912,  1911,  1955,  1957,  1961,  1962,
    1967,  1974,  1974,  1983,  1982,  2046,  2047,  2053,  2055,  2059,
    2060,  2063,  2082,  2091,  2090,  2108,  2109,  2110,  2117,  2133,
    2134,  2135,  2145,  2146,  2147,  2148,  2149,  2150,  2154,  2172,
    2173,  2174,  2185,  2186,  2187,  2188,  2189,  2190,  2191,  2192,
    2193,  2194,  2195,  2196,  2197,  2198,  2199,  2200,  2201,  2202,
    2203,  2204,  2205,  2206,  2207,  2208,  2209,  2210,  2211,  2212,
    2213,  2214,  2215,  2216,  2217,  2218,  2219,  2220,  2221,  2222,
    2223,  2224,  2225,  2226,  2227,  2229,  2234,  2238,  2243,  2249,
    2258,  2259,  2261,  2266,  2273,  2274,  2275,  2276,  2277,  2278,
    2279,  2280,  2281,  2282,  2283,  2284,  2289,  2292,  2295,  2298,
    2301,  2307,  2313,  2318,  2318,  2328,  2327,  2368,  2369,  2373,
    2381,  2380,  2426,  2425,  2468,  2469,  2478,  2483,  2490,  2497,
    2507,  2508,  2512,  2520,  2521,  2525,  2534,  2535,  2536,  2544,
    2545,  2549,  2550,  2554,  2560,  2567,  2568
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
  "OPENSQUARE", "DOT", "ARROW", "ASSIGN_POW", "ASSIGN_SHL", "ASSIGN_SHR",
  "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD", "ASSIGN_DIV",
  "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_TO", "COMMA",
  "QUESTION", "OR", "AND", "NOT", "LE", "GE", "LT", "GT", "NEQ", "EEQ",
  "PROVIDES", "OP_NOTIN", "OP_IN", "HASNT", "HAS", "DIESIS", "ATSIGN",
  "CAP_CAP", "VBAR_VBAR", "AMPER_AMPER", "MINUS", "PLUS", "PERCENT",
  "SLASH", "STAR", "POW", "SHR", "SHL", "TILDE", "NEG", "AMPER", "BANG",
  "DECREMENT", "INCREMENT", "DOLLAR", "$accept", "input", "body", "line",
  "toplevel_statement", "INTNUM_WITH_MINUS", "load_statement", "statement",
  "base_statement", "assignment_def_list", "def_statement",
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
  "expression_list", "listpar_expression_list", "symbol_list",
  "expression_pair_list", 0
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
     365,   366,   367
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   113,   114,   115,   115,   116,   116,   116,   117,   117,
     117,   117,   117,   117,   117,   117,   117,   117,   118,   118,
     119,   119,   119,   120,   120,   120,   120,   120,   120,   121,
     121,   121,   121,   121,   121,   121,   121,   121,   121,   121,
     121,   121,   121,   121,   121,   121,   121,   121,   121,   121,
     121,   122,   122,   123,   123,   125,   124,   124,   126,   126,
     126,   127,   127,   127,   129,   128,   128,   130,   130,   131,
     131,   132,   132,   133,   132,   134,   134,   136,   135,   137,
     137,   138,   138,   139,   139,   140,   140,   140,   142,   141,
     143,   141,   141,   141,   144,   144,   145,   145,   145,   145,
     146,   146,   147,   147,   147,   147,   147,   147,   148,   150,
     149,   149,   149,   152,   151,   151,   151,   154,   153,   153,
     153,   156,   155,   157,   157,   158,   158,   158,   159,   160,
     159,   161,   159,   162,   159,   163,   159,   164,   165,   164,
     166,   166,   166,   167,   167,   168,   168,   169,   169,   169,
     169,   169,   171,   170,   172,   172,   173,   173,   173,   174,
     175,   174,   176,   174,   177,   174,   178,   174,   179,   179,
     180,   180,   180,   181,   181,   181,   182,   183,   182,   184,
     184,   185,   185,   186,   186,   187,   188,   188,   188,   188,
     188,   189,   189,   190,   190,   191,   191,   192,   192,   193,
     194,   193,   193,   195,   196,   195,   197,   198,   198,   198,
     199,   200,   201,   200,   202,   200,   203,   203,   204,   204,
     205,   205,   206,   206,   206,   206,   207,   207,   207,   208,
     208,   208,   209,   209,   210,   210,   211,   211,   212,   212,
     213,   213,   214,   214,   214,   215,   215,   216,   216,   217,
     217,   218,   218,   218,   219,   219,   219,   221,   220,   222,
     222,   223,   223,   224,   223,   225,   225,   226,   226,   227,
     228,   228,   229,   229,   230,   230,   230,   231,   231,   232,
     232,   232,   232,   234,   233,   235,   235,   236,   236,   236,
     237,   237,   237,   237,   239,   238,   240,   240,   241,   241,
     241,   242,   242,   244,   243,   245,   245,   246,   246,   247,
     247,   247,   247,   249,   248,   250,   250,   250,   251,   252,
     252,   252,   253,   253,   253,   253,   253,   253,   254,   255,
     255,   255,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   257,   257,   257,   257,
     257,   258,   258,   259,   258,   261,   260,   262,   262,   262,
     264,   263,   266,   265,   267,   267,   268,   268,   268,   268,
     269,   269,   269,   270,   270,   270,   271,   271,   271,   272,
     272,   273,   273,   274,   274,   275,   275
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     3,     3,     3,     2,     2,
       4,     6,     1,     1,     1,     1,     1,     1,     1,     1,
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
       3,     3,     1,     3,     3,     3,     1,     3,     3,     3,
       1,     3,     3,     3,     3,     4,     3,     2,     3,     2,
       3,     0,     1,     3,     2,     3,     2,     0,     8,     3,
       2,     0,     3,     0,     5,     0,     2,     1,     3,     2,
       0,     3,     1,     3,     1,     1,     1,     0,     2,     1,
       1,     1,     1,     0,     7,     5,     4,     0,     3,     3,
       1,     2,     3,     4,     0,     7,     0,     2,     1,     4,
       2,     1,     1,     0,     8,     2,     2,     0,     2,     1,
       1,     1,     1,     0,     4,     1,     3,     3,     1,     2,
       3,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     2,     3,     2,     2,
       2,     2,     3,     3,     3,     3,     3,     3,     3,     2,
       3,     3,     3,     3,     3,     2,     2,     2,     2,     1,
       1,     1,     1,     1,     1,     2,     1,     4,     5,     3,
       1,     1,     3,     5,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     4,     4,     5,
       7,     4,     3,     0,     6,     0,     6,     4,     3,     2,
       0,     6,     0,     5,     1,     2,     5,     5,     4,     3,
       2,     3,     3,     2,     3,     3,     3,     3,     4,     1,
       3,     1,     2,     1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    24,   325,   326,   328,   327,
     322,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   331,   330,     0,     0,     0,     0,     0,     0,   313,
     412,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     410,     0,     0,    59,     0,   323,   324,   108,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       4,     5,     8,    13,    23,    32,    33,    55,     0,    37,
      64,     0,    38,    39,    34,    47,    48,    49,    35,   121,
      36,   152,    40,    41,   177,    42,    10,   211,     0,     0,
      45,    46,    14,    15,    16,     9,    17,     0,   251,    11,
      50,    12,    44,    43,   332,   329,   333,   429,   381,   372,
     370,   371,   369,   373,   376,   374,   380,     0,    28,     6,
       0,     0,     0,     0,   405,     0,     0,    83,     0,    85,
       0,     0,     0,     0,   433,     0,     0,     0,     0,     0,
       0,     0,   429,     0,     0,   179,     0,     0,     0,     0,
     257,     0,   303,     0,   319,     0,     0,     0,     0,     0,
       0,     0,     0,   372,     0,     0,     0,   247,   249,     0,
       0,     0,   229,   232,     0,     0,   232,     0,     0,     0,
       0,     0,   240,     0,   206,     0,     0,     0,     0,   423,
     431,     0,    62,   294,     0,     0,   420,     0,   429,     0,
       0,   359,     0,   105,     0,   368,   367,   334,     0,   103,
       0,   346,   351,   349,   366,   365,    81,     0,     0,     0,
      57,    81,    66,   125,   156,    81,     0,    81,   212,   214,
     198,     0,     0,     0,   252,     0,   251,     0,    29,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   350,
     348,   375,     0,     0,    54,    53,     0,     0,    60,    63,
      58,    61,    84,    87,    86,    68,    70,    67,    69,    93,
       0,     0,   124,   123,     7,   155,   154,   175,     0,     0,
     180,   176,   196,   195,    27,     0,    26,     0,   321,   320,
     318,     0,   315,     0,   210,   414,   208,     0,    22,    20,
      21,   221,   220,   228,     0,   248,   250,   225,   222,     0,
     231,   230,     0,   235,     0,   234,   239,     0,   238,     0,
      25,     0,   207,   211,   211,   101,   100,   425,   424,   432,
       0,   395,   396,     0,   426,     0,     0,   422,   421,     0,
     427,     0,   107,   104,   106,   102,     0,     0,     0,     0,
       0,     0,   216,   218,     0,    81,     0,   202,   204,     0,
     256,   254,     0,     0,     0,   246,   402,     0,     0,     0,
     379,   389,   393,   394,   392,   391,   390,   388,   387,   386,
     385,   384,   382,   419,     0,   358,   357,   356,   355,   354,
     353,   347,   352,   364,   363,   362,   361,   360,   343,   342,
     341,   336,   335,   339,   338,   337,   340,   345,   344,     0,
     430,     0,    51,   434,     0,     0,   174,     0,     0,   207,
     277,   265,     0,     0,     0,   307,   314,     0,   415,     0,
       0,     0,     0,     0,   362,   233,   233,    18,   242,   243,
       0,   244,   241,   409,     0,    81,    81,   296,   398,   397,
       0,   435,   428,     0,     0,    82,     0,     0,     0,    73,
      72,    77,     0,   128,     0,     0,   126,     0,   138,     0,
     159,     0,     0,   157,     0,     0,   182,   183,    81,   217,
     219,     0,     0,   215,     0,   200,     0,   253,   245,   255,
     403,   401,     0,   377,     0,   418,     0,    30,     0,     0,
      92,    88,    90,   173,   260,     0,   287,     0,   306,   270,
     266,   267,   305,   287,   317,   316,   209,   413,   227,   226,
     224,   223,    19,   408,     0,     0,     0,     0,     0,   399,
       0,    56,     0,    75,     0,     0,     0,    81,    81,   127,
       0,   151,   149,   147,   148,     0,   145,   142,     0,     0,
     158,     0,   171,   172,     0,   168,     0,     0,   186,   193,
     194,     0,     0,   191,     0,   184,     0,   197,     0,     0,
       0,   199,   203,     0,   378,   383,   417,   416,     0,    52,
       0,     0,   263,   262,   279,     0,     0,     0,     0,     0,
     280,   278,   282,   281,     0,   259,     0,   269,     0,   309,
     310,   312,   311,     0,   308,   407,   406,   411,   298,     0,
       0,   297,     0,   436,    76,    80,    79,    65,     0,     0,
     133,   135,     0,   129,   131,     0,   122,    81,     0,   139,
     164,   166,   160,   162,   170,   153,   190,     0,   188,     0,
       0,   178,   213,   205,     0,   404,    31,     0,     0,     0,
      96,     0,     0,    97,    98,    99,    91,     0,     0,   283,
       0,     0,   290,     0,     0,     0,   275,   276,     0,   272,
     274,   268,     0,   301,     0,   302,   300,   295,   400,    78,
      81,     0,   150,    81,     0,   146,     0,   144,    81,     0,
      81,     0,   169,   187,   192,     0,   201,     0,   109,     0,
       0,   113,     0,     0,   117,     0,     0,    95,   264,     0,
     211,     0,   289,   291,   288,     0,   258,   271,     0,   304,
       0,     0,   136,     0,   132,     0,   167,     0,   163,   189,
     112,    81,   111,   116,    81,   115,   120,    81,   119,    89,
     286,    81,     0,   292,     0,   273,   299,     0,     0,     0,
       0,   285,   293,     0,     0,     0,     0,   110,   114,   118,
     284
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    60,    61,   574,    62,   485,    64,   121,
      65,    66,   216,    67,    68,    69,   221,    70,    71,   488,
     567,   489,   490,   568,   491,   376,    72,    73,    74,   610,
     611,   681,   682,    75,    76,    77,   683,   761,   684,   764,
     685,   767,    78,   223,    79,   378,   496,   713,   714,   710,
     711,   497,   579,   498,   659,   575,   576,    80,   224,    81,
     379,   503,   720,   721,   718,   719,   584,   585,    82,    83,
     225,    84,   505,   506,   507,   508,   592,   593,    85,    86,
      87,   600,    88,   514,    89,   325,   326,   227,   385,   386,
     228,   229,    90,    91,    92,    93,   174,    94,   178,    95,
     181,   182,    96,    97,    98,   235,   236,    99,   315,   450,
     451,   687,   454,   540,   541,   627,   698,   699,   536,   621,
     622,   740,   623,   624,   694,   100,   360,   557,   641,   706,
     101,   317,   455,   543,   634,   102,   156,   321,   322,   103,
     104,   105,   106,   107,   108,   109,   603,   110,   185,   353,
     111,   186,   112,   157,   327,   113,   114,   115,   116,   117,
     191,   135,   200
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -345
static const yytype_int16 yypact[] =
{
    -345,    21,   756,  -345,    28,  -345,  -345,  -345,  -345,  -345,
    -345,    85,   304,  3191,   300,   437,  3256,   306,  3321,    57,
    3386,  -345,  -345,  3451,   166,  3516,   338,   377,   473,  -345,
    -345,   359,  3581,   431,   281,  3646,   213,   435,   438,   123,
    -345,  3711,  5060,    88,    90,  -345,  -345,  -345,  5320,  4963,
    5320,   593,  5320,  5320,  5320,  3061,  5320,  5320,  5320,   373,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  2996,  -345,
    -345,  2996,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,   129,  2996,   116,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,    77,   200,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  4378,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,   378,  -345,  -345,
     165,    29,   140,    25,  -345,  4152,   215,  -345,   222,  -345,
     223,   130,  4212,   264,  -345,    23,   270,  4431,   283,   290,
    4484,   316,  5834,    55,   319,  -345,  2996,   349,  4537,   355,
    -345,   396,  -345,   406,  -345,  4590,   429,    89,   439,   449,
     461,   462,  5834,   468,   469,   381,   161,  -345,  -345,   488,
    4643,   489,  -345,  -345,    78,   493,   494,   421,   503,   504,
     434,    95,  -345,   508,  -345,   162,   162,   510,  4696,  -345,
    5834,  3126,  -345,  -345,  5569,  5125,  -345,   455,  5397,    83,
      93,  6069,   515,  -345,   109,  4009,  4009,   237,   516,  -345,
     119,   237,   237,   237,  -345,  -345,  -345,   519,   521,   164,
    -345,  -345,  -345,  -345,  -345,  -345,   282,  -345,  -345,  -345,
    -345,   523,    81,   524,  -345,   124,   196,   133,  -345,  5190,
    4990,   525,  5320,  5320,  5320,  5320,  5320,  5320,  5320,  5320,
    5320,  5320,  5320,  5320,  3776,  5320,  5320,  5320,  5320,  5320,
    5320,  5320,  5320,   529,  5320,  5320,  5320,  5320,  5320,  5320,
    5320,  5320,  5320,  5320,  5320,  5320,  5320,  5320,  5320,  -345,
    -345,  -345,  5320,  5320,  -345,  -345,   531,  5320,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,
     531,  3841,  -345,  -345,  -345,  -345,  -345,  -345,   527,  5320,
    -345,  -345,  -345,  -345,  -345,   276,  -345,   187,  -345,  -345,
    -345,   134,  -345,   540,  -345,   467,  -345,   482,  -345,  -345,
    -345,  -345,  -345,  -345,   287,  -345,  -345,  -345,  -345,  3906,
    -345,  -345,   541,  -345,   549,  -345,  -345,    47,  -345,   550,
    -345,   554,   552,   129,   129,  -345,  -345,  -345,  -345,  5834,
     556,  -345,  -345,  5622,  -345,  5255,  5320,  -345,  -345,   500,
    -345,  5320,  -345,  -345,  -345,  -345,  1764,  1428,   314,   317,
    1540,   182,  -345,  -345,  1876,  -345,  2996,  -345,  -345,    33,
    -345,  -345,   559,   565,   135,  -345,  -345,    63,  5320,  5456,
    -345,  5834,  5834,  5834,  5834,  5834,  5834,  5834,  5834,  5834,
    5834,  5834,  5881,  -345,  4092,  6022,  6069,  6116,  6116,  6116,
    6116,  6116,  6116,  -345,  4009,  4009,  4009,  4009,   271,   271,
     384,   502,   502,   252,   252,   252,   511,   237,   237,  4265,
    5928,   487,  5834,  -345,   566,  4325,  -345,   191,   571,   552,
    -345,   542,   573,   572,   574,  -345,  -345,   453,  -345,   552,
    5320,   577,   579,   583,  5338,  -345,   584,  -345,  -345,  -345,
     587,  -345,  -345,  -345,    84,  -345,  -345,  -345,  -345,  -345,
    5515,  5834,  -345,  5675,   585,  -345,   459,  3971,   601,  -345,
    -345,  -345,   589,  -345,    11,   299,  -345,   602,  -345,   611,
    -345,    73,   608,  -345,    67,   609,   594,  -345,  -345,  -345,
    -345,   620,  1988,  -345,   567,  -345,   269,  -345,  -345,  -345,
    -345,  -345,  5728,  -345,  5320,  -345,  4036,  -345,  5320,  5320,
    -345,  -345,  -345,  -345,  -345,   104,    54,   623,  -345,   568,
     555,  -345,  -345,    59,  -345,  -345,  -345,  5834,  -345,  -345,
    -345,  -345,  -345,  -345,   628,  2100,  2212,   424,  5320,  -345,
    5320,  -345,   630,  -345,   631,  4749,   632,  -345,  -345,  -345,
     277,  -345,  -345,  -345,   560,    72,  -345,  -345,   634,   278,
    -345,   284,  -345,  -345,   112,  -345,   635,   637,  -345,  -345,
    -345,   531,    24,  -345,   639,  -345,  1652,  -345,   641,   599,
     592,  -345,  -345,   595,  -345,   570,  -345,  5975,   192,  5834,
     868,  2996,  -345,  -345,  -345,   576,   652,   651,   653,    49,
    -345,  -345,  -345,  -345,   649,  -345,   497,  -345,   572,  -345,
    -345,  -345,  -345,   654,  -345,  -345,  -345,  -345,  -345,    96,
     658,  -345,  5781,  5834,  -345,  -345,  -345,  -345,  2324,  1428,
    -345,  -345,     6,  -345,  -345,    18,  -345,  -345,  2996,  -345,
    -345,  -345,  -345,  -345,   457,  -345,  -345,   659,  -345,   464,
     531,  -345,  -345,  -345,   661,  -345,  -345,   303,   333,   446,
    -345,   656,   868,  -345,  -345,  -345,  -345,   610,  5320,  -345,
     591,   664,  -345,   663,   197,   667,  -345,  -345,   -25,  -345,
    -345,  -345,   668,  -345,   534,  -345,  -345,  -345,  -345,  -345,
    -345,  2996,  -345,  -345,  2996,  -345,  2436,  -345,  -345,  2996,
    -345,  2996,  -345,  -345,  -345,   669,  -345,   670,  -345,  2996,
     672,  -345,  2996,   674,  -345,  2996,   675,  -345,  -345,  4802,
     129,  5320,  -345,  -345,  -345,     8,  -345,  -345,   497,  -345,
     201,   980,  -345,  1092,  -345,  1204,  -345,  1316,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,
    -345,  -345,  4855,  -345,   673,  -345,  -345,  2548,  2660,  2772,
    2884,  -345,  -345,   678,   679,   680,   681,  -345,  -345,  -345,
    -345
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -345,  -345,  -345,  -345,  -345,  -344,  -345,    -2,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,    36,
    -345,  -345,  -345,  -345,  -345,   -18,  -345,  -345,  -345,  -345,
    -345,     7,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,
    -345,   309,  -345,  -345,  -345,  -345,    35,  -345,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,    30,  -345,  -345,
    -345,  -345,  -345,  -345,   190,  -345,  -345,    32,  -345,  -319,
    -345,  -345,  -345,  -345,  -345,  -227,   234,  -306,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,   660,  -345,  -345,  -345,
    -345,   357,  -345,  -345,  -345,   -89,  -345,  -345,  -345,  -345,
    -345,  -345,   247,  -345,    79,  -345,  -345,   -40,  -345,  -345,
     167,  -345,   168,   170,  -345,  -345,  -345,  -345,  -345,   -36,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,   258,  -345,
    -275,   -10,  -345,   -12,   -14,   684,  -345,  -345,  -345,   532,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,    12,
    -345,  -345,  -345
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -406
static const yytype_int16 yytable[] =
{
      63,   125,   122,   471,   132,   389,   137,   134,   140,   237,
     467,   142,   570,   148,   773,   467,   155,   571,   572,   573,
     162,     3,   467,   170,   571,   572,   573,   668,   288,   188,
     190,   118,   285,   747,   515,   143,   194,   198,   201,   142,
     205,   206,   207,   142,   211,   212,   213,   475,   476,   215,
     691,   467,   748,   468,   469,   692,   308,   614,   138,   462,
     615,   199,   629,   204,   520,   615,   220,   210,   587,   222,
     588,   589,   289,   590,   581,   653,  -170,   582,   233,   583,
    -251,   341,   388,   234,   367,   553,   230,   324,   119,   774,
     323,   516,   616,   281,   369,   324,   193,   616,   348,   703,
     300,   669,   617,   618,   470,   612,   286,   617,   618,   470,
     459,   281,   373,   301,   670,   662,   470,   231,   281,   654,
    -170,   521,   375,   281,   183,   474,   281,   391,   281,   184,
     693,   309,   283,   295,   281,   192,   395,   456,   519,  -207,
     283,   281,   554,   368,   311,   470,   619,   394,   281,   655,
    -170,   619,  -207,   370,  -251,   342,   281,   591,  -207,   663,
     283,   459,   613,   351,   335,   183,  -207,   144,   284,   145,
     371,   704,   349,   705,   281,   232,   281,   296,   226,   359,
     281,   459,  -405,   363,   281,   509,   283,   281,   452,   664,
    -265,   281,   281,   281,   533,   676,   283,   281,   281,   281,
     744,   392,   234,   377,   703,   393,   234,   380,   336,   384,
     392,   457,   392,   146,   171,   287,   172,   620,   292,   173,
     453,   352,   535,  -405,   630,   293,   294,   142,   399,   510,
     401,   402,   403,   404,   405,   406,   407,   408,   409,   410,
     411,   412,   414,   415,   416,   417,   418,   419,   420,   421,
     422,   397,   424,   425,   426,   427,   428,   429,   430,   431,
     432,   433,   434,   435,   436,   437,   438,   299,   283,   283,
     439,   440,   601,   302,   745,   442,   441,   448,   705,  -261,
     650,   657,   166,   381,   167,   382,   304,   660,   461,   445,
     443,     6,     7,   305,     9,    10,   239,   142,   240,   241,
     577,   126,  -141,   127,   727,   120,   728,   133,   712,  -261,
       8,   239,     8,   240,   241,   492,   602,   493,   499,   307,
     500,   447,   310,  -137,   651,   658,  -137,   464,   168,   383,
     239,   661,   240,   241,   730,   449,   731,   494,   495,   149,
     501,   495,    45,    46,   150,   281,  -141,   279,   280,   281,
     729,   700,   312,   480,   481,   276,   277,   278,   314,   483,
     158,  -140,   279,   280,  -140,   159,   160,   512,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   214,   151,     8,
     732,   279,   280,   152,   513,   281,   522,   281,   281,   281,
     281,   281,   281,   281,   281,   281,   281,   281,   281,   316,
     281,   281,   281,   281,   281,   281,   281,   281,   281,   318,
     281,   281,   281,   281,   281,   281,   281,   281,   281,   281,
     281,   281,   281,   281,   281,   281,   281,   638,   281,   750,
     639,   281,   164,   640,   771,   320,   175,   165,   128,   179,
     129,   176,   328,   239,   180,   240,   241,   733,   547,   734,
     281,   130,   329,   282,   544,   283,   334,   555,   556,   320,
     562,   582,   563,   583,   330,   331,   281,   281,   589,   281,
     590,   332,   333,   700,   153,   565,   154,     6,     7,     8,
       9,    10,   271,   272,   273,   274,   275,   276,   277,   278,
     596,   337,   340,   735,   279,   280,   343,  -236,   344,    21,
      22,     6,     7,   696,     9,    10,   345,   346,   281,   347,
      30,   350,   142,   355,   607,   364,   142,   609,   372,   374,
     149,   124,   151,    40,   697,    42,   387,   390,    45,    46,
     446,   400,    48,   281,    49,   423,   605,     8,     6,     7,
     608,     9,    10,   458,   459,   460,   642,   465,   643,   648,
     649,   281,    45,    46,    50,   466,   180,   473,   324,   477,
     482,   239,   529,   240,   241,   517,    52,    53,   518,   530,
     239,    54,   240,   241,   534,   453,   538,   542,   539,    56,
     548,   667,   549,    57,    58,    59,   550,  -237,   561,    45,
      46,   552,   569,   281,   202,   281,   203,     6,     7,     8,
       9,    10,   273,   274,   275,   276,   277,   278,   680,   686,
     566,   578,   279,   280,   580,   277,   278,   586,   594,    21,
      22,   279,   280,   597,   504,   599,   625,   626,   281,   281,
      30,   635,   628,   644,   645,   647,   652,   656,   665,   716,
     666,   124,   671,    40,   672,    42,   673,   283,    45,    46,
     674,   688,    48,   675,    49,   689,   717,   184,   695,   690,
     725,   707,   723,   702,   726,   736,   741,   742,   738,   743,
     746,   749,   759,   760,    50,   763,   739,   766,   769,   782,
     680,   787,   788,   789,   790,   709,    52,    53,   502,   737,
     715,    54,   751,   546,   722,   753,   595,   177,   537,    56,
     755,   724,   757,    57,    58,    59,   472,   701,   775,   752,
     631,   632,   754,   633,   776,   545,   163,   756,   354,   758,
       0,     0,     0,     0,     0,   281,     0,   762,     0,   772,
     765,     0,     0,   768,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   777,     0,     0,   778,     0,     0,   779,
       0,     0,     0,   780,     0,     0,    -2,     4,   281,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,    19,
       0,    20,    21,    22,    23,    24,     0,    25,    26,     0,
      27,    28,    29,    30,     0,    31,    32,    33,    34,    35,
      36,    37,    38,     0,    39,     0,    40,    41,    42,    43,
      44,    45,    46,    47,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,     0,
      55,     0,    56,     0,     0,     0,    57,    58,    59,     4,
       0,     5,     6,     7,     8,     9,    10,   -94,    12,    13,
      14,    15,     0,    16,     0,     0,    17,   677,   678,   679,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     217,     0,   218,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   219,     0,    40,    41,
      42,    43,    44,    45,    46,    47,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,     0,
       0,     0,    55,     0,    56,     0,     0,     0,    57,    58,
      59,     4,     0,     5,     6,     7,     8,     9,    10,  -134,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,  -134,  -134,    20,    21,    22,    23,    24,
       0,    25,   217,     0,   218,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,  -134,   219,     0,
      40,    41,    42,    43,    44,    45,    46,    47,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,    51,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,     0,     0,    54,     0,
       0,     0,     0,     0,    55,     0,    56,     0,     0,     0,
      57,    58,    59,     4,     0,     5,     6,     7,     8,     9,
      10,  -130,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,  -130,  -130,    20,    21,    22,
      23,    24,     0,    25,   217,     0,   218,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,  -130,
     219,     0,    40,    41,    42,    43,    44,    45,    46,    47,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
      54,     0,     0,     0,     0,     0,    55,     0,    56,     0,
       0,     0,    57,    58,    59,     4,     0,     5,     6,     7,
       8,     9,    10,  -165,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,  -165,  -165,    20,
      21,    22,    23,    24,     0,    25,   217,     0,   218,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,  -165,   219,     0,    40,    41,    42,    43,    44,    45,
      46,    47,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
       0,     0,    54,     0,     0,     0,     0,     0,    55,     0,
      56,     0,     0,     0,    57,    58,    59,     4,     0,     5,
       6,     7,     8,     9,    10,  -161,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,  -161,
    -161,    20,    21,    22,    23,    24,     0,    25,   217,     0,
     218,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,  -161,   219,     0,    40,    41,    42,    43,
      44,    45,    46,    47,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,     0,
      55,     0,    56,     0,     0,     0,    57,    58,    59,     4,
       0,     5,     6,     7,     8,     9,    10,   -71,    12,    13,
      14,    15,     0,    16,   486,   487,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     217,     0,   218,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   219,     0,    40,    41,
      42,    43,    44,    45,    46,    47,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,     0,
       0,     0,    55,     0,    56,     0,     0,     0,    57,    58,
      59,     4,     0,     5,     6,     7,     8,     9,    10,  -181,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
     504,    25,   217,     0,   218,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   219,     0,
      40,    41,    42,    43,    44,    45,    46,    47,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,    51,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,     0,     0,    54,     0,
       0,     0,     0,     0,    55,     0,    56,     0,     0,     0,
      57,    58,    59,     4,     0,     5,     6,     7,     8,     9,
      10,  -185,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,  -185,    25,   217,     0,   218,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     219,     0,    40,    41,    42,    43,    44,    45,    46,    47,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
      54,     0,     0,     0,     0,     0,    55,     0,    56,     0,
       0,     0,    57,    58,    59,     4,     0,     5,     6,     7,
       8,     9,    10,   484,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   217,     0,   218,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   219,     0,    40,    41,    42,    43,    44,    45,
      46,    47,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
       0,     0,    54,     0,     0,     0,     0,     0,    55,     0,
      56,     0,     0,     0,    57,    58,    59,     4,     0,     5,
       6,     7,     8,     9,    10,   511,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   217,     0,
     218,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   219,     0,    40,    41,    42,    43,
      44,    45,    46,    47,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,     0,
      55,     0,    56,     0,     0,     0,    57,    58,    59,     4,
       0,     5,     6,     7,     8,     9,    10,   598,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     217,     0,   218,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   219,     0,    40,    41,
      42,    43,    44,    45,    46,    47,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,     0,
       0,     0,    55,     0,    56,     0,     0,     0,    57,    58,
      59,     4,     0,     5,     6,     7,     8,     9,    10,   636,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   217,     0,   218,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   219,     0,
      40,    41,    42,    43,    44,    45,    46,    47,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,    51,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,     0,     0,    54,     0,
       0,     0,     0,     0,    55,     0,    56,     0,     0,     0,
      57,    58,    59,     4,     0,     5,     6,     7,     8,     9,
      10,   637,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   217,     0,   218,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     219,     0,    40,    41,    42,    43,    44,    45,    46,    47,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
      54,     0,     0,     0,     0,     0,    55,     0,    56,     0,
       0,     0,    57,    58,    59,     4,     0,     5,     6,     7,
       8,     9,    10,   -74,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   217,     0,   218,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   219,     0,    40,    41,    42,    43,    44,    45,
      46,    47,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
       0,     0,    54,     0,     0,     0,     0,     0,    55,     0,
      56,     0,     0,     0,    57,    58,    59,     4,     0,     5,
       6,     7,     8,     9,    10,  -143,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   217,     0,
     218,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   219,     0,    40,    41,    42,    43,
      44,    45,    46,    47,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,     0,
      55,     0,    56,     0,     0,     0,    57,    58,    59,     4,
       0,     5,     6,     7,     8,     9,    10,   783,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     217,     0,   218,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   219,     0,    40,    41,
      42,    43,    44,    45,    46,    47,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,     0,
       0,     0,    55,     0,    56,     0,     0,     0,    57,    58,
      59,     4,     0,     5,     6,     7,     8,     9,    10,   784,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   217,     0,   218,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   219,     0,
      40,    41,    42,    43,    44,    45,    46,    47,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,    51,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,     0,     0,    54,     0,
       0,     0,     0,     0,    55,     0,    56,     0,     0,     0,
      57,    58,    59,     4,     0,     5,     6,     7,     8,     9,
      10,   785,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   217,     0,   218,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     219,     0,    40,    41,    42,    43,    44,    45,    46,    47,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
      54,     0,     0,     0,     0,     0,    55,     0,    56,     0,
       0,     0,    57,    58,    59,     4,     0,     5,     6,     7,
       8,     9,    10,   786,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   217,     0,   218,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   219,     0,    40,    41,    42,    43,    44,    45,
      46,    47,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
       0,     0,    54,     0,     0,     0,     0,     0,    55,     0,
      56,     0,     0,     0,    57,    58,    59,     4,     0,     5,
       6,     7,     8,     9,    10,     0,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   217,     0,
     218,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   219,     0,    40,    41,    42,    43,
      44,    45,    46,    47,     0,    48,     0,    49,     0,     0,
       0,     0,   208,     0,   209,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,    21,    22,    52,
      53,     0,     0,     0,    54,     0,     0,     0,    30,     0,
      55,     0,    56,     0,     0,     0,    57,    58,    59,   124,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,     0,     0,   357,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,     0,
       0,    57,    58,    59,   124,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,   358,    49,     0,     0,
       0,     0,   123,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,    52,
      53,     0,     0,     0,    54,     0,     0,     0,    30,     0,
       0,     0,    56,     0,     0,     0,    57,    58,    59,   124,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,     0,     0,   131,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,     0,
       0,    57,    58,    59,   124,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,   136,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,    52,
      53,     0,     0,     0,    54,     0,     0,     0,    30,     0,
       0,     0,    56,     0,     0,     0,    57,    58,    59,   124,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,     0,     0,   139,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,     0,
       0,    57,    58,    59,   124,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,   141,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,    52,
      53,     0,     0,     0,    54,     0,     0,     0,    30,     0,
       0,     0,    56,     0,     0,     0,    57,    58,    59,   124,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,     0,     0,   147,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,     0,
       0,    57,    58,    59,   124,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,   161,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,    52,
      53,     0,     0,     0,    54,     0,     0,     0,    30,     0,
       0,     0,    56,     0,     0,     0,    57,    58,    59,   124,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,     0,     0,   169,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,     0,
       0,    57,    58,    59,   124,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,   187,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,    52,
      53,     0,     0,     0,    54,     0,     0,     0,    30,     0,
       0,     0,    56,     0,     0,     0,    57,    58,    59,   124,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,     0,     0,   413,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,     0,
       0,    57,    58,    59,   124,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,   444,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,    52,
      53,     0,     0,     0,    54,     0,     0,     0,    30,     0,
       0,     0,    56,     0,     0,     0,    57,    58,    59,   124,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,     0,     0,   463,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,     0,
       0,    57,    58,    59,   124,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,   564,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    21,    22,    52,
      53,     0,     0,     0,    54,     0,     0,     0,    30,     0,
       0,     0,    56,     0,     0,     0,    57,    58,    59,   124,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,     0,     0,   606,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,   239,    54,
     240,   241,     0,    30,     0,     0,     0,    56,     0,     0,
       0,    57,    58,    59,   124,     0,    40,     0,    42,     0,
       0,    45,    46,   525,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,     0,     0,    50,     0,   279,
     280,     0,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,   526,
       0,     0,    56,     0,     0,     0,    57,    58,    59,     0,
       0,   239,     0,   240,   241,   290,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,     0,     0,
     254,   255,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,     0,   291,
       0,     0,   279,   280,     0,     0,     0,     0,     0,     0,
       0,   239,     0,   240,   241,   297,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,     0,     0,
     254,   255,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,     0,   298,
       0,     0,   279,   280,     0,     0,     0,     0,   527,     0,
       0,   239,     0,   240,   241,     0,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,     0,     0,
     254,   255,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,     0,     0,
       0,     0,   279,   280,   239,     0,   240,   241,   531,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,   528,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,   532,     0,     0,   279,   280,     0,     0,     0,
       0,   238,     0,     0,   239,     0,   240,   241,     0,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,     0,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,     0,     0,   303,   279,   280,   239,     0,   240,
     241,     0,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,     0,     0,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,     0,     0,   306,   279,   280,
     239,     0,   240,   241,     0,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,     0,     0,   254,
     255,   256,     0,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     0,     0,     0,
     313,   279,   280,   239,     0,   240,   241,     0,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   252,   253,
       0,     0,   254,   255,   256,     0,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,     0,     0,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
       0,     0,     0,   319,   279,   280,   239,     0,   240,   241,
       0,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,     0,     0,   254,   255,   256,     0,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
       0,     0,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,     0,     0,     0,   338,   279,   280,   239,
       0,   240,   241,     0,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   253,     0,     0,   254,   255,
     256,     0,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,     0,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,     0,     0,     0,   356,
     279,   280,   239,     0,   240,   241,     0,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,     0,
       0,   254,   255,   256,     0,   257,   258,   259,   260,   261,
     262,   263,   264,   339,   266,   267,     0,     0,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,     0,
       0,     0,   646,   279,   280,   239,     0,   240,   241,     0,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,     0,     0,   254,   255,   256,     0,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,     0,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,     0,     0,     0,   770,   279,   280,   239,     0,
     240,   241,     0,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,     0,     0,   254,   255,   256,
       0,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,     0,     0,     0,   781,   279,
     280,   239,     0,   240,   241,     0,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,     0,     0,
     254,   255,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,     0,     0,
       0,     0,   279,   280,   239,     0,   240,   241,     0,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,     0,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,     0,     0,     0,   279,   280,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     6,     7,     8,     9,    10,     0,
      30,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     195,   124,     0,    40,     0,    42,    21,    22,    45,    46,
       0,     0,    48,   196,    49,     0,   197,    30,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   195,   124,     0,
      40,     0,    42,     0,    50,    45,    46,     0,     0,    48,
       0,    49,     0,     0,     0,     0,    52,    53,     0,     0,
       0,    54,     0,     0,     6,     7,     8,     9,    10,    56,
       0,    50,     0,    57,    58,    59,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,    21,    22,    54,     0,
       0,     0,   398,     0,     0,     0,    56,    30,     0,     0,
      57,    58,    59,     0,     0,     0,     0,     0,   124,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
     189,    49,     0,     0,     0,     0,     0,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,    52,    53,     0,     0,     0,    54,     0,
       0,     0,    30,     0,     0,     0,    56,     0,     0,     0,
      57,    58,    59,   124,     0,    40,     0,    42,     0,     0,
      45,    46,     0,     0,    48,   362,    49,     0,     0,     0,
       0,     0,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,    52,    53,
       0,     0,     0,    54,     0,     0,     0,    30,     0,     0,
       0,    56,     0,     0,     0,    57,    58,    59,   124,     0,
      40,     0,    42,     0,     0,    45,    46,     0,   396,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,    52,    53,     0,     0,     0,    54,     0,
       0,     0,    30,     0,     0,     0,    56,     0,     0,     0,
      57,    58,    59,   124,     0,    40,     0,    42,     0,     0,
      45,    46,     0,     0,    48,   479,    49,     0,     0,     0,
       0,     0,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
       0,   551,     0,     0,     0,     0,    21,    22,    52,    53,
       0,     0,     0,    54,     0,     0,     0,    30,     0,     0,
       0,    56,     0,     0,     0,    57,    58,    59,   124,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   239,     0,   240,
     241,    50,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,     0,     0,    54,     0,
       0,     0,     0,     0,     0,     0,    56,     0,     0,     0,
      57,    58,    59,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   365,     0,     0,     0,   279,   280,
       0,     0,     0,     0,     0,     0,   239,     0,   240,   241,
     366,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,     0,     0,   254,   255,   256,     0,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
       0,     0,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   365,     0,     0,     0,   279,   280,     0,
       0,     0,     0,     0,     0,   239,   523,   240,   241,     0,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,     0,     0,   254,   255,   256,     0,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,     0,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   558,     0,     0,     0,   279,   280,     0,     0,
       0,     0,     0,     0,   239,   559,   240,   241,     0,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,     0,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,     0,     0,     0,   279,   280,   361,   239,     0,
     240,   241,     0,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,     0,     0,   254,   255,   256,
       0,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,     0,     0,     0,     0,   279,
     280,   239,   478,   240,   241,     0,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,     0,     0,
     254,   255,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,     0,     0,
       0,     0,   279,   280,   239,     0,   240,   241,   560,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,     0,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,     0,     0,     0,   279,   280,   239,   604,   240,
     241,     0,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,     0,     0,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,     0,     0,     0,   279,   280,
     239,   708,   240,   241,     0,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,     0,     0,   254,
     255,   256,     0,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     0,     0,     0,
       0,   279,   280,   239,     0,   240,   241,     0,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   252,   253,
       0,     0,   254,   255,   256,     0,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,     0,     0,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     239,     0,   240,   241,   279,   280,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   253,     0,   524,   254,
     255,   256,     0,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   239,     0,   240,
     241,   279,   280,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   239,     0,   240,   241,   279,   280,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   239,     0,   240,   241,   279,   280,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   239,     0,
     240,   241,   279,   280,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   239,     0,   240,   241,   279,
     280,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   263,   264,   265,   266,   267,     0,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,     0,     0,     0,     0,   279,   280
};

static const yytype_int16 yycheck[] =
{
       2,    13,    12,   347,    16,   232,    18,    17,    20,    98,
       4,    23,     1,    25,     6,     4,    28,     6,     7,     8,
      32,     0,     4,    35,     6,     7,     8,     3,     3,    41,
      42,     3,     3,    58,     1,    23,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,   353,   354,    59,
       1,     4,    77,     6,     7,     6,     1,     3,     1,   334,
       6,    49,     3,    51,     1,     6,    68,    55,     1,    71,
       3,     4,    47,     6,     1,     3,     3,     4,     1,     6,
       3,     3,     1,     6,     1,     1,    88,     6,     3,    81,
       1,    58,    38,   107,     1,     6,     6,    38,     3,     3,
      77,    77,    48,    49,    98,     1,    77,    48,    49,    98,
      77,   125,     3,    90,    90,     3,    98,     1,   132,    47,
      47,    58,     3,   137,     1,   352,   140,     3,   142,     6,
      81,    76,    77,     3,   148,    47,     3,     3,     3,    58,
      77,   155,    58,    60,   146,    98,    92,   236,   162,    77,
      77,    92,    63,    60,    77,    77,   170,    90,    77,    47,
      77,    77,    58,     1,     3,     1,    77,     1,     3,     3,
      77,    75,    77,    77,   188,    59,   190,    47,    49,   191,
     194,    77,    59,   195,   198,     3,    77,   201,     1,    77,
       3,   205,   206,   207,     3,     3,    77,   211,   212,   213,
       3,    77,     6,   221,     3,     9,     6,   225,    47,   227,
      77,    77,    77,    47,     1,    75,     3,   536,     3,     6,
      33,    59,   449,    59,   543,     3,     3,   239,   240,    47,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   239,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     3,    77,    77,
     282,   283,     3,     3,    77,   287,   286,     1,    77,     3,
       3,     3,     1,     1,     3,     3,     3,     3,     1,   301,
     300,     4,     5,     3,     7,     8,    59,   309,    61,    62,
       1,     1,     3,     3,     1,     1,     3,     1,   652,    33,
       6,    59,     6,    61,    62,     1,    47,     3,     1,     3,
       3,   309,     3,     9,    47,    47,     9,   339,    47,    47,
      59,    47,    61,    62,     1,    59,     3,    23,    24,     1,
      23,    24,    55,    56,     6,   359,    47,   110,   111,   363,
      47,   626,     3,   365,   366,   103,   104,   105,     3,   371,
       1,    47,   110,   111,    47,     6,     7,   385,    97,    98,
      99,   100,   101,   102,   103,   104,   105,     4,     1,     6,
      47,   110,   111,     6,   386,   399,   398,   401,   402,   403,
     404,   405,   406,   407,   408,   409,   410,   411,   412,     3,
     414,   415,   416,   417,   418,   419,   420,   421,   422,     3,
     424,   425,   426,   427,   428,   429,   430,   431,   432,   433,
     434,   435,   436,   437,   438,   439,   440,     3,   442,   704,
       6,   445,     1,     9,   740,     6,     1,     6,     1,     1,
       3,     6,     3,    59,     6,    61,    62,     1,   460,     3,
     464,    14,     3,    75,     1,    77,    75,   475,   476,     6,
       1,     4,     3,     6,     3,     3,   480,   481,     4,   483,
       6,     3,     3,   748,     1,   487,     3,     4,     5,     6,
       7,     8,    98,    99,   100,   101,   102,   103,   104,   105,
     508,     3,     3,    47,   110,   111,     3,     3,    77,    26,
      27,     4,     5,     6,     7,     8,     3,     3,   522,    75,
      37,     3,   524,     3,   526,    60,   528,   529,     3,     3,
       1,    48,     1,    50,    27,    52,     3,     3,    55,    56,
       3,     6,    59,   547,    61,     6,   524,     6,     4,     5,
     528,     7,     8,     3,    77,    63,   558,     6,   560,   567,
     568,   565,    55,    56,    81,     6,     6,     3,     6,     3,
      60,    59,    75,    61,    62,     6,    93,    94,     3,     3,
      59,    98,    61,    62,     3,    33,     3,     3,     6,   106,
       3,   591,     3,   110,   111,   112,     3,     3,     3,    55,
      56,     4,     3,   607,     1,   609,     3,     4,     5,     6,
       7,     8,   100,   101,   102,   103,   104,   105,   610,   611,
       9,     9,   110,   111,     3,   104,   105,     9,     9,    26,
      27,   110,   111,     3,    30,    58,     3,    59,   642,   643,
      37,     3,    77,     3,     3,     3,    76,     3,     3,   657,
       3,    48,     3,    50,     3,    52,    47,    77,    55,    56,
      58,    75,    59,    58,    61,     3,   658,     6,     9,     6,
     670,     3,     3,     9,     3,     9,    75,     3,    58,     6,
       3,     3,     3,     3,    81,     3,   688,     3,     3,     6,
     682,     3,     3,     3,     3,   649,    93,    94,   379,   682,
     655,    98,   710,   459,   664,   713,   506,    37,   451,   106,
     718,   669,   720,   110,   111,   112,   349,   628,   748,   711,
     543,   543,   714,   543,   750,   457,    32,   719,   186,   721,
      -1,    -1,    -1,    -1,    -1,   739,    -1,   729,    -1,   741,
     732,    -1,    -1,   735,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   761,    -1,    -1,   764,    -1,    -1,   767,
      -1,    -1,    -1,   771,    -1,    -1,     0,     1,   772,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    39,    40,    41,    42,    43,
      44,    45,    46,    -1,    48,    -1,    50,    51,    52,    53,
      54,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,
     104,    -1,   106,    -1,    -1,    -1,   110,   111,   112,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    19,    20,    21,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    54,    55,    56,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,
      -1,    -1,   104,    -1,   106,    -1,    -1,    -1,   110,   111,
     112,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    47,    48,    -1,
      50,    51,    52,    53,    54,    55,    56,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    -1,    -1,   104,    -1,   106,    -1,    -1,    -1,
     110,   111,   112,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    24,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    47,
      48,    -1,    50,    51,    52,    53,    54,    55,    56,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,    -1,
      -1,    -1,   110,   111,   112,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    47,    48,    -1,    50,    51,    52,    53,    54,    55,
      56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,
      24,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    47,    48,    -1,    50,    51,    52,    53,
      54,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,
     104,    -1,   106,    -1,    -1,    -1,   110,   111,   112,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    16,    17,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    54,    55,    56,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,
      -1,    -1,   104,    -1,   106,    -1,    -1,    -1,   110,   111,
     112,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      30,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    54,    55,    56,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    -1,    -1,   104,    -1,   106,    -1,    -1,    -1,
     110,   111,   112,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    30,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    54,    55,    56,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,    -1,
      -1,    -1,   110,   111,   112,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    54,    55,
      56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      54,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,
     104,    -1,   106,    -1,    -1,    -1,   110,   111,   112,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    54,    55,    56,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,
      -1,    -1,   104,    -1,   106,    -1,    -1,    -1,   110,   111,
     112,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    54,    55,    56,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    -1,    -1,   104,    -1,   106,    -1,    -1,    -1,
     110,   111,   112,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    54,    55,    56,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,    -1,
      -1,    -1,   110,   111,   112,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    54,    55,
      56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      54,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,
     104,    -1,   106,    -1,    -1,    -1,   110,   111,   112,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    54,    55,    56,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,
      -1,    -1,   104,    -1,   106,    -1,    -1,    -1,   110,   111,
     112,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    54,    55,    56,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    -1,    -1,   104,    -1,   106,    -1,    -1,    -1,
     110,   111,   112,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    54,    55,    56,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,    -1,
      -1,    -1,   110,   111,   112,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    54,    55,
      56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,     1,    -1,     3,
       4,     5,     6,     7,     8,    -1,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      54,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,     1,    -1,     3,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    85,    -1,    -1,    -1,    -1,    -1,    26,    27,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    37,    -1,
     104,    -1,   106,    -1,    -1,    -1,   110,   111,   112,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    60,    61,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    37,    -1,
      -1,    -1,   106,    -1,    -1,    -1,   110,   111,   112,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    37,    -1,
      -1,    -1,   106,    -1,    -1,    -1,   110,   111,   112,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    37,    -1,
      -1,    -1,   106,    -1,    -1,    -1,   110,   111,   112,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    37,    -1,
      -1,    -1,   106,    -1,    -1,    -1,   110,   111,   112,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    37,    -1,
      -1,    -1,   106,    -1,    -1,    -1,   110,   111,   112,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    37,    -1,
      -1,    -1,   106,    -1,    -1,    -1,   110,   111,   112,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    37,    -1,
      -1,    -1,   106,    -1,    -1,    -1,   110,   111,   112,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    93,    94,    -1,    -1,    59,    98,
      61,    62,    -1,    37,    -1,    -1,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,     1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,    81,    -1,   110,
     111,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    47,
      -1,    -1,   106,    -1,    -1,    -1,   110,   111,   112,    -1,
      -1,    59,    -1,    61,    62,     3,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    47,
      -1,    -1,   110,   111,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    59,    -1,    61,    62,     3,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    47,
      -1,    -1,   110,   111,    -1,    -1,    -1,    -1,     3,    -1,
      -1,    59,    -1,    61,    62,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
      -1,    -1,   110,   111,    59,    -1,    61,    62,     3,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    77,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    47,    -1,    -1,   110,   111,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    59,    -1,    61,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    -1,    -1,     3,   110,   111,    59,    -1,    61,
      62,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    -1,    -1,     3,   110,   111,
      59,    -1,    61,    62,    -1,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    78,
      79,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    -1,    -1,    -1,
       3,   110,   111,    59,    -1,    61,    62,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    78,    79,    80,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,    -1,     3,   110,   111,    59,    -1,    61,    62,
      -1,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    78,    79,    80,    -1,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,    -1,     3,   110,   111,    59,
      -1,    61,    62,    -1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    78,    79,
      80,    -1,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,    -1,    -1,     3,
     110,   111,    59,    -1,    61,    62,    -1,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      -1,    78,    79,    80,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    -1,    -1,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,    -1,
      -1,    -1,     3,   110,   111,    59,    -1,    61,    62,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    78,    79,    80,    -1,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,    -1,     3,   110,   111,    59,    -1,
      61,    62,    -1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    78,    79,    80,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,    -1,     3,   110,
     111,    59,    -1,    61,    62,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
      -1,    -1,   110,   111,    59,    -1,    61,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    -1,    -1,    -1,   110,   111,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      47,    48,    -1,    50,    -1,    52,    26,    27,    55,    56,
      -1,    -1,    59,    60,    61,    -1,    63,    37,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    47,    48,    -1,
      50,    -1,    52,    -1,    81,    55,    56,    -1,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,     4,     5,     6,     7,     8,   106,
      -1,    81,    -1,   110,   111,   112,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    26,    27,    98,    -1,
      -1,    -1,   102,    -1,    -1,    -1,   106,    37,    -1,    -1,
     110,   111,   112,    -1,    -1,    -1,    -1,    -1,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      60,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    37,    -1,    -1,    -1,   106,    -1,    -1,    -1,
     110,   111,   112,    48,    -1,    50,    -1,    52,    -1,    -1,
      55,    56,    -1,    -1,    59,    60,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    37,    -1,    -1,
      -1,   106,    -1,    -1,    -1,   110,   111,   112,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    58,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    37,    -1,    -1,    -1,   106,    -1,    -1,    -1,
     110,   111,   112,    48,    -1,    50,    -1,    52,    -1,    -1,
      55,    56,    -1,    -1,    59,    60,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    -1,    -1,    26,    27,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    37,    -1,    -1,
      -1,   106,    -1,    -1,    -1,   110,   111,   112,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    59,    -1,    61,
      62,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   106,    -1,    -1,    -1,
     110,   111,   112,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    47,    -1,    -1,    -1,   110,   111,
      -1,    -1,    -1,    -1,    -1,    -1,    59,    -1,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    78,    79,    80,    -1,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    47,    -1,    -1,    -1,   110,   111,    -1,
      -1,    -1,    -1,    -1,    -1,    59,    60,    61,    62,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    78,    79,    80,    -1,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    47,    -1,    -1,    -1,   110,   111,    -1,    -1,
      -1,    -1,    -1,    -1,    59,    60,    61,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    -1,    -1,    -1,   110,   111,    58,    59,    -1,
      61,    62,    -1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    78,    79,    80,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,    -1,    -1,   110,
     111,    59,    60,    61,    62,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
      -1,    -1,   110,   111,    59,    -1,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    -1,    -1,    -1,   110,   111,    59,    60,    61,
      62,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    -1,    -1,    -1,   110,   111,
      59,    60,    61,    62,    -1,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    78,
      79,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    -1,    -1,    -1,
      -1,   110,   111,    59,    -1,    61,    62,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    78,    79,    80,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      59,    -1,    61,    62,   110,   111,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    75,    -1,    77,    78,
      79,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    59,    -1,    61,
      62,   110,   111,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    59,    -1,    61,    62,   110,   111,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    59,    -1,    61,    62,   110,   111,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    59,    -1,
      61,    62,   110,   111,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    59,    -1,    61,    62,   110,
     111,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,    -1,    -1,   110,   111
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   114,   115,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    45,    46,    48,
      50,    51,    52,    53,    54,    55,    56,    57,    59,    61,
      81,    85,    93,    94,    98,   104,   106,   110,   111,   112,
     116,   117,   119,   120,   121,   123,   124,   126,   127,   128,
     130,   131,   139,   140,   141,   146,   147,   148,   155,   157,
     170,   172,   181,   182,   184,   191,   192,   193,   195,   197,
     205,   206,   207,   208,   210,   212,   215,   216,   217,   220,
     238,   243,   248,   252,   253,   254,   255,   256,   257,   258,
     260,   263,   265,   268,   269,   270,   271,   272,     3,     3,
       1,   122,   254,     1,    48,   256,     1,     3,     1,     3,
      14,     1,   256,     1,   254,   274,     1,   256,     1,     1,
     256,     1,   256,   272,     1,     3,    47,     1,   256,     1,
       6,     1,     6,     1,     3,   256,   249,   266,     1,     6,
       7,     1,   256,   258,     1,     6,     1,     3,    47,     1,
     256,     1,     3,     6,   209,     1,     6,   209,   211,     1,
       6,   213,   214,     1,     6,   261,   264,     1,   256,    60,
     256,   273,    47,     6,   256,    47,    60,    63,   256,   272,
     275,   256,     1,     3,   272,   256,   256,   256,     1,     3,
     272,   256,   256,   256,     4,   254,   125,    32,    34,    48,
     120,   129,   120,   156,   171,   183,    49,   200,   203,   204,
     120,     1,    59,     1,     6,   218,   219,   218,     3,    59,
      61,    62,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    78,    79,    80,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   110,
     111,   257,    75,    77,     3,     3,    77,    75,     3,    47,
       3,    47,     3,     3,     3,     3,    47,     3,    47,     3,
      77,    90,     3,     3,     3,     3,     3,     3,     1,    76,
       3,   120,     3,     3,     3,   221,     3,   244,     3,     3,
       6,   250,   251,     1,     6,   198,   199,   267,     3,     3,
       3,     3,     3,     3,    75,     3,    47,     3,     3,    90,
       3,     3,    77,     3,    77,     3,     3,    75,     3,    77,
       3,     1,    59,   262,   262,     3,     3,     1,    60,   256,
     239,    58,    60,   256,    60,    47,    63,     1,    60,     1,
      60,    77,     3,     3,     3,     3,   138,   138,   158,   173,
     138,     1,     3,    47,   138,   201,   202,     3,     1,   198,
       3,     3,    77,     9,   218,     3,    58,   272,   102,   256,
       6,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,     1,   256,   256,   256,   256,   256,   256,
     256,   256,   256,     6,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   254,   256,   254,     1,   256,     3,   272,     1,    59,
     222,   223,     1,    33,   225,   245,     3,    77,     3,    77,
      63,     1,   253,     1,   256,     6,     6,     4,     6,     7,
      98,   118,   214,     3,   198,   200,   200,     3,    60,    60,
     256,   256,    60,   256,     9,   120,    16,    17,   132,   134,
     135,   137,     1,     3,    23,    24,   159,   164,   166,     1,
       3,    23,   164,   174,    30,   185,   186,   187,   188,     3,
      47,     9,   138,   120,   196,     1,    58,     6,     3,     3,
       1,    58,   256,    60,    77,     1,    47,     3,    77,    75,
       3,     3,    47,     3,     3,   198,   231,   225,     3,     6,
     226,   227,     3,   246,     1,   251,   199,   256,     3,     3,
       3,     3,     4,     1,    58,   138,   138,   240,    47,    60,
      63,     3,     1,     3,     1,   256,     9,   133,   136,     3,
       1,     6,     7,     8,   118,   168,   169,     1,     9,   165,
       3,     1,     4,     6,   179,   180,     9,     1,     3,     4,
       6,    90,   189,   190,     9,   187,   138,     3,     9,    58,
     194,     3,    47,   259,    60,   272,     1,   256,   272,   256,
     142,   143,     1,    58,     3,     6,    38,    48,    49,    92,
     192,   232,   233,   235,   236,     3,    59,   228,    77,     3,
     192,   233,   235,   236,   247,     3,     9,     9,     3,     6,
       9,   241,   256,   256,     3,     3,     3,     3,   138,   138,
       3,    47,    76,     3,    47,    77,     3,     3,    47,   167,
       3,    47,     3,    47,    77,     3,     3,   254,     3,    77,
      90,     3,     3,    47,    58,    58,     3,    19,    20,    21,
     120,   144,   145,   149,   151,   153,   120,   224,    75,     3,
       6,     1,     6,    81,   237,     9,     6,    27,   229,   230,
     253,   227,     9,     3,    75,    77,   242,     3,    60,   132,
     162,   163,   118,   160,   161,   169,   138,   120,   177,   178,
     175,   176,   180,     3,   190,   254,     3,     1,     3,    47,
       1,     3,    47,     1,     3,    47,     9,   144,    58,   256,
     234,    75,     3,     6,     3,    77,     3,    58,    77,     3,
     253,   138,   120,   138,   120,   138,   120,   138,   120,     3,
       3,   150,   120,     3,   152,   120,     3,   154,   120,     3,
       3,   200,   256,     6,    81,   230,   242,   138,   138,   138,
     138,     3,     6,     9,     9,     9,     9,     3,     3,     3,
       3
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
#line 206 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); }
    break;

  case 7:
#line 207 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); }
    break;

  case 8:
#line 212 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      }
    break;

  case 10:
#line 218 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 11:
#line 223 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 12:
#line 228 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 13:
#line 233 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 19:
#line 245 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); }
    break;

  case 20:
#line 250 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      }
    break;

  case 21:
#line 256 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      }
    break;

  case 22:
#line 262 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      }
    break;

  case 23:
#line 269 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); }
    break;

  case 24:
#line 270 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; }
    break;

  case 25:
#line 271 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; }
    break;

  case 26:
#line 272 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; }
    break;

  case 27:
#line 273 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; }
    break;

  case 28:
#line 274 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;}
    break;

  case 29:
#line 279 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); }
    break;

  case 30:
#line 281 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      }
    break;

  case 31:
#line 287 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 323 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   }
    break;

  case 52:
#line 329 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   }
    break;

  case 53:
#line 338 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; }
    break;

  case 54:
#line 340 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); }
    break;

  case 55:
#line 344 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 56:
#line 351 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 57:
#line 358 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 58:
#line 366 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 59:
#line 367 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 60:
#line 368 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; }
    break;

  case 61:
#line 372 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 62:
#line 373 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 63:
#line 374 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 64:
#line 378 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
    break;

  case 65:
#line 386 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 66:
#line 393 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      }
    break;

  case 67:
#line 403 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 68:
#line 404 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; }
    break;

  case 69:
#line 408 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 70:
#line 409 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 73:
#line 416 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
    break;

  case 76:
#line 426 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); }
    break;

  case 77:
#line 431 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
    break;

  case 79:
#line 443 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 80:
#line 444 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; }
    break;

  case 82:
#line 449 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   }
    break;

  case 83:
#line 456 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 465 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      }
    break;

  case 85:
#line 473 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 483 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 492 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      }
    break;

  case 88:
#line 501 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 517 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 90:
#line 525 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 541 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 551 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 93:
#line 556 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 96:
#line 568 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      }
    break;

  case 100:
#line 582 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 595 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      }
    break;

  case 102:
#line 603 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 103:
#line 607 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 104:
#line 613 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 105:
#line 619 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      }
    break;

  case 106:
#line 626 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 107:
#line 631 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 108:
#line 640 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   }
    break;

  case 109:
#line 649 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 661 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 111:
#line 663 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 672 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
    break;

  case 113:
#line 676 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 688 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 115:
#line 689 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 698 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); }
    break;

  case 117:
#line 702 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 716 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 119:
#line 718 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 727 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
    break;

  case 121:
#line 731 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 122:
#line 739 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 123:
#line 748 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 124:
#line 750 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 127:
#line 759 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); }
    break;

  case 129:
#line 765 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 131:
#line 775 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 132:
#line 783 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 133:
#line 787 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 799 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 809 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 138:
#line 818 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 832 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); }
    break;

  case 144:
#line 836 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 147:
#line 848 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }
    break;

  case 148:
#line 857 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 869 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 880 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 891 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 911 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 153:
#line 919 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 154:
#line 928 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 155:
#line 930 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 158:
#line 939 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); }
    break;

  case 160:
#line 945 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 162:
#line 955 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 163:
#line 964 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 164:
#line 968 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 980 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 990 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 171:
#line 1004 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1016 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1037 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_adecl), (yyvsp[(2) - (5)].fal_adecl) );
      }
    break;

  case 174:
#line 1041 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      }
    break;

  case 175:
#line 1045 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; }
    break;

  case 176:
#line 1053 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   }
    break;

  case 177:
#line 1060 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }
    break;

  case 178:
#line 1070 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
    break;

  case 180:
#line 1079 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); }
    break;

  case 186:
#line 1099 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1117 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1137 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 189:
#line 1147 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1158 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }
    break;

  case 193:
#line 1171 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1183 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1205 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 196:
#line 1206 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; }
    break;

  case 197:
#line 1218 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 198:
#line 1224 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 200:
#line 1233 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 201:
#line 1234 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 202:
#line 1237 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
    break;

  case 204:
#line 1242 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 205:
#line 1243 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 206:
#line 1250 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
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
#line 1311 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1328 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 213:
#line 1334 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 214:
#line 1339 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 215:
#line 1345 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 217:
#line 1354 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); }
    break;

  case 219:
#line 1359 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
    break;

  case 220:
#line 1369 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 221:
#line 1372 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; }
    break;

  case 222:
#line 1381 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 223:
#line 1388 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1403 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      }
    break;

  case 225:
#line 1409 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      }
    break;

  case 226:
#line 1421 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1431 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      }
    break;

  case 228:
#line 1436 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      }
    break;

  case 229:
#line 1448 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1457 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      }
    break;

  case 231:
#line 1464 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      }
    break;

  case 232:
#line 1472 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      }
    break;

  case 233:
#line 1477 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      }
    break;

  case 234:
#line 1485 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = 0;
      }
    break;

  case 235:
#line 1489 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 236:
#line 1497 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->imported(true);
      }
    break;

  case 237:
#line 1502 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->imported(true);
      }
    break;

  case 238:
#line 1514 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 239:
#line 1519 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 242:
#line 1532 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 243:
#line 1536 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 244:
#line 1540 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 245:
#line 1554 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 246:
#line 1561 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 248:
#line 1569 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); }
    break;

  case 250:
#line 1573 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); }
    break;

  case 252:
#line 1579 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         }
    break;

  case 253:
#line 1583 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         }
    break;

  case 256:
#line 1592 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   }
    break;

  case 257:
#line 1603 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 258:
#line 1637 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 260:
#line 1665 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 263:
#line 1673 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 264:
#line 1674 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 269:
#line 1691 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 270:
#line 1724 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; }
    break;

  case 271:
#line 1729 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
   }
    break;

  case 272:
#line 1735 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 273:
#line 1736 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 275:
#line 1742 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      }
    break;

  case 276:
#line 1750 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 280:
#line 1760 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 281:
#line 1763 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 283:
#line 1785 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 284:
#line 1809 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 285:
#line 1820 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 286:
#line 1842 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 289:
#line 1872 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   }
    break;

  case 290:
#line 1879 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      }
    break;

  case 291:
#line 1887 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      }
    break;

  case 292:
#line 1893 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      }
    break;

  case 293:
#line 1899 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      }
    break;

  case 294:
#line 1912 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 295:
#line 1946 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 299:
#line 1963 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 300:
#line 1968 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 303:
#line 1983 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      }
    break;

  case 304:
#line 2023 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( !cd->inheritance().empty() )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 306:
#line 2048 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 310:
#line 2060 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 311:
#line 2063 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 313:
#line 2091 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 314:
#line 2096 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 317:
#line 2111 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 318:
#line 2118 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 319:
#line 2133 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 320:
#line 2134 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 321:
#line 2135 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 322:
#line 2145 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 323:
#line 2146 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 324:
#line 2147 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 325:
#line 2148 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 326:
#line 2149 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 327:
#line 2150 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 328:
#line 2155 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 330:
#line 2173 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 331:
#line 2174 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); }
    break;

  case 334:
#line 2187 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 335:
#line 2188 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 336:
#line 2189 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 337:
#line 2190 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 338:
#line 2191 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 339:
#line 2192 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 340:
#line 2193 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 341:
#line 2194 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 342:
#line 2195 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 343:
#line 2196 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 344:
#line 2197 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 345:
#line 2198 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 346:
#line 2199 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 347:
#line 2200 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 348:
#line 2201 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 349:
#line 2202 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 350:
#line 2203 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 351:
#line 2204 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 352:
#line 2205 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 353:
#line 2206 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 354:
#line 2207 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 355:
#line 2208 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 356:
#line 2209 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 357:
#line 2210 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 358:
#line 2211 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 359:
#line 2212 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 360:
#line 2213 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 361:
#line 2214 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 362:
#line 2215 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 363:
#line 2216 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 364:
#line 2217 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 365:
#line 2218 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 366:
#line 2219 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 367:
#line 2220 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 368:
#line 2221 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 375:
#line 2229 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 376:
#line 2234 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 377:
#line 2238 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 378:
#line 2243 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 379:
#line 2249 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 382:
#line 2261 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 383:
#line 2266 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 384:
#line 2273 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 385:
#line 2274 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 386:
#line 2275 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 387:
#line 2276 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 388:
#line 2277 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 389:
#line 2278 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 390:
#line 2279 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 391:
#line 2280 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 392:
#line 2281 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 393:
#line 2282 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 394:
#line 2283 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 395:
#line 2284 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 396:
#line 2289 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 397:
#line 2292 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 398:
#line 2295 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 399:
#line 2298 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 400:
#line 2301 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 401:
#line 2308 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 402:
#line 2314 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 403:
#line 2318 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 404:
#line 2319 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 405:
#line 2328 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
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

  case 406:
#line 2362 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 408:
#line 2370 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 409:
#line 2374 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 410:
#line 2381 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
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

  case 411:
#line 2414 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 412:
#line 2426 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
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

  case 413:
#line 2458 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            COMPILER->checkLocalUndefined();
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 415:
#line 2470 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      }
    break;

  case 416:
#line 2479 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 417:
#line 2484 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 418:
#line 2491 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 419:
#line 2498 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 420:
#line 2507 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 421:
#line 2509 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 422:
#line 2513 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 423:
#line 2520 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 424:
#line 2522 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 425:
#line 2526 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 426:
#line 2534 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 427:
#line 2535 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 428:
#line 2537 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 429:
#line 2544 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 430:
#line 2545 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 431:
#line 2549 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 432:
#line 2550 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (2)].fal_adecl)->pushBack( (yyvsp[(2) - (2)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (2)].fal_adecl); }
    break;

  case 433:
#line 2554 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 434:
#line 2560 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 435:
#line 2567 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 436:
#line 2568 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;


/* Line 1267 of yacc.c.  */
#line 6276 "/home/user/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2572 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


