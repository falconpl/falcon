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
     OP_TO = 332,
     COMMA = 333,
     QUESTION = 334,
     OR = 335,
     AND = 336,
     NOT = 337,
     LE = 338,
     GE = 339,
     LT = 340,
     GT = 341,
     NEQ = 342,
     EEQ = 343,
     PROVIDES = 344,
     OP_NOTIN = 345,
     OP_IN = 346,
     HASNT = 347,
     HAS = 348,
     DIESIS = 349,
     ATSIGN = 350,
     CAP_CAP = 351,
     VBAR_VBAR = 352,
     AMPER_AMPER = 353,
     MINUS = 354,
     PLUS = 355,
     PERCENT = 356,
     SLASH = 357,
     STAR = 358,
     POW = 359,
     SHR = 360,
     SHL = 361,
     TILDE = 362,
     NEG = 363,
     AMPER = 364,
     BANG = 365,
     DECREMENT = 366,
     INCREMENT = 367,
     DOLLAR = 368
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
#define OP_TO 332
#define COMMA 333
#define QUESTION 334
#define OR 335
#define AND 336
#define NOT 337
#define LE 338
#define GE 339
#define LT 340
#define GT 341
#define NEQ 342
#define EEQ 343
#define PROVIDES 344
#define OP_NOTIN 345
#define OP_IN 346
#define HASNT 347
#define HAS 348
#define DIESIS 349
#define ATSIGN 350
#define CAP_CAP 351
#define VBAR_VBAR 352
#define AMPER_AMPER 353
#define MINUS 354
#define PLUS 355
#define PERCENT 356
#define SLASH 357
#define STAR 358
#define POW 359
#define SHR 360
#define SHL 361
#define TILDE 362
#define NEG 363
#define AMPER 364
#define BANG 365
#define DECREMENT 366
#define INCREMENT 367
#define DOLLAR 368




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
#line 387 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 400 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   6244

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  114
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  164
/* YYNRULES -- Number of rules.  */
#define YYNRULES  450
/* YYNRULES -- Number of states.  */
#define YYNSTATES  822

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   368

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
     105,   106,   107,   108,   109,   110,   111,   112,   113
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
     795,   800,   805,   812,   819,   823,   825,   829,   833,   837,
     839,   843,   847,   851,   855,   860,   864,   867,   871,   874,
     878,   879,   881,   885,   888,   892,   895,   896,   905,   909,
     912,   913,   917,   918,   924,   925,   928,   930,   934,   937,
     938,   942,   944,   948,   950,   952,   954,   955,   958,   960,
     962,   964,   966,   967,   975,   981,   986,   987,   991,   995,
     997,  1000,  1004,  1009,  1010,  1018,  1019,  1022,  1024,  1029,
    1032,  1034,  1036,  1037,  1046,  1049,  1052,  1053,  1056,  1058,
    1060,  1062,  1064,  1065,  1070,  1072,  1076,  1080,  1082,  1085,
    1089,  1093,  1095,  1097,  1099,  1101,  1103,  1105,  1107,  1109,
    1111,  1113,  1115,  1117,  1120,  1123,  1126,  1130,  1134,  1138,
    1142,  1146,  1150,  1154,  1158,  1162,  1166,  1170,  1174,  1177,
    1181,  1184,  1187,  1190,  1193,  1197,  1201,  1205,  1209,  1213,
    1217,  1221,  1224,  1228,  1232,  1236,  1240,  1244,  1247,  1250,
    1253,  1256,  1258,  1260,  1262,  1264,  1266,  1268,  1271,  1273,
    1278,  1284,  1288,  1290,  1292,  1296,  1302,  1306,  1310,  1314,
    1318,  1322,  1326,  1330,  1334,  1338,  1342,  1346,  1350,  1354,
    1359,  1364,  1370,  1378,  1383,  1387,  1388,  1395,  1396,  1403,
    1408,  1412,  1415,  1416,  1423,  1424,  1430,  1432,  1435,  1441,
    1447,  1452,  1456,  1459,  1463,  1467,  1470,  1474,  1478,  1482,
    1486,  1491,  1493,  1497,  1499,  1503,  1504,  1506,  1508,  1512,
    1516
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     115,     0,    -1,   116,    -1,    -1,   116,   117,    -1,   118,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   120,    -1,
     213,    -1,   193,    -1,   221,    -1,   244,    -1,   239,    -1,
     121,    -1,   208,    -1,   209,    -1,   211,    -1,   216,    -1,
       4,    -1,    99,     4,    -1,    39,     6,     3,    -1,    39,
       7,     3,    -1,    39,     1,     3,    -1,   122,    -1,     3,
      -1,    48,     1,     3,    -1,    34,     1,     3,    -1,    32,
       1,     3,    -1,     1,     3,    -1,   257,     3,    -1,   273,
      76,   257,     3,    -1,   273,    76,   257,    78,   273,     3,
      -1,   124,    -1,   125,    -1,   142,    -1,   156,    -1,   171,
      -1,   129,    -1,   140,    -1,   141,    -1,   182,    -1,   183,
      -1,   192,    -1,   253,    -1,   249,    -1,   206,    -1,   207,
      -1,   147,    -1,   148,    -1,   149,    -1,   255,    76,   257,
      -1,   123,    78,   255,    76,   257,    -1,    10,   123,     3,
      -1,    10,     1,     3,    -1,    -1,   127,   126,   139,     9,
       3,    -1,   128,   121,    -1,    11,   257,     3,    -1,    53,
      -1,    11,     1,     3,    -1,    11,   257,    47,    -1,    53,
      47,    -1,    11,     1,    47,    -1,    -1,   131,   130,   139,
     133,     9,     3,    -1,   132,   121,    -1,    15,   257,     3,
      -1,    15,     1,     3,    -1,    15,   257,    47,    -1,    15,
       1,    47,    -1,    -1,   136,    -1,    -1,   135,   134,   139,
      -1,    16,     3,    -1,    16,     1,     3,    -1,    -1,   138,
     137,   139,   133,    -1,    17,   257,     3,    -1,    17,     1,
       3,    -1,    -1,   139,   121,    -1,    12,     3,    -1,    12,
       1,     3,    -1,    13,     3,    -1,    13,    14,     3,    -1,
      13,     1,     3,    -1,    -1,    18,   276,    91,   257,     3,
     143,   145,     9,     3,    -1,    -1,    18,   276,    91,   257,
      47,   144,   121,    -1,    18,   276,    91,     1,     3,    -1,
      18,     1,     3,    -1,    -1,   146,   145,    -1,   121,    -1,
     150,    -1,   152,    -1,   154,    -1,    51,   257,     3,    -1,
      51,     1,     3,    -1,   105,   273,     3,    -1,   105,     3,
      -1,    86,   273,     3,    -1,    86,     3,    -1,   105,     1,
       3,    -1,    86,     1,     3,    -1,    57,    -1,    -1,    19,
       3,   151,   139,     9,     3,    -1,    19,    47,   121,    -1,
      19,     1,     3,    -1,    -1,    20,     3,   153,   139,     9,
       3,    -1,    20,    47,   121,    -1,    20,     1,     3,    -1,
      -1,    21,     3,   155,   139,     9,     3,    -1,    21,    47,
     121,    -1,    21,     1,     3,    -1,    -1,   158,   157,   159,
     165,     9,     3,    -1,    22,   257,     3,    -1,    22,     1,
       3,    -1,    -1,   159,   160,    -1,   159,     1,     3,    -1,
       3,    -1,    -1,    23,   169,     3,   161,   139,    -1,    -1,
      23,   169,    47,   162,   121,    -1,    -1,    23,     1,     3,
     163,   139,    -1,    -1,    23,     1,    47,   164,   121,    -1,
      -1,    -1,   167,   166,   168,    -1,    -1,    24,    -1,    24,
       1,    -1,     3,   139,    -1,    47,   121,    -1,   170,    -1,
     169,    78,   170,    -1,     8,    -1,   119,    -1,     7,    -1,
     119,    77,   119,    -1,     6,    -1,    -1,   173,   172,   174,
     165,     9,     3,    -1,    25,   257,     3,    -1,    25,     1,
       3,    -1,    -1,   174,   175,    -1,   174,     1,     3,    -1,
       3,    -1,    -1,    23,   180,     3,   176,   139,    -1,    -1,
      23,   180,    47,   177,   121,    -1,    -1,    23,     1,     3,
     178,   139,    -1,    -1,    23,     1,    47,   179,   121,    -1,
     181,    -1,   180,    78,   181,    -1,    -1,     4,    -1,     6,
      -1,    28,   273,    77,   273,     3,    -1,    28,   273,     1,
       3,    -1,    28,     1,     3,    -1,    29,    47,   121,    -1,
      -1,   185,   184,   139,   186,     9,     3,    -1,    29,     3,
      -1,    29,     1,     3,    -1,    -1,   187,    -1,   188,    -1,
     187,   188,    -1,   189,   139,    -1,    30,     3,    -1,    30,
      91,   255,     3,    -1,    30,   190,     3,    -1,    30,   190,
      91,   255,     3,    -1,    30,     1,     3,    -1,   191,    -1,
     190,    78,   191,    -1,     4,    -1,     6,    -1,    31,   257,
       3,    -1,    31,     1,     3,    -1,   194,   201,   139,     9,
       3,    -1,   196,   121,    -1,   198,    59,   199,    58,     3,
      -1,    -1,   198,    59,   199,     1,   195,    58,     3,    -1,
     198,     1,     3,    -1,   198,    59,   199,    58,    47,    -1,
      -1,   198,    59,     1,   197,    58,    47,    -1,    48,     6,
      -1,    -1,   200,    -1,   199,    78,   200,    -1,     6,    -1,
      -1,    -1,   204,   202,   139,     9,     3,    -1,    -1,   205,
     203,   121,    -1,    49,     3,    -1,    49,     1,     3,    -1,
      49,    47,    -1,    49,     1,    47,    -1,    40,   259,     3,
      -1,    40,     1,     3,    -1,    43,   257,     3,    -1,    43,
     257,    91,   257,     3,    -1,    43,   257,    91,     1,     3,
      -1,    43,     1,     3,    -1,    41,     6,    76,   254,     3,
      -1,    41,     6,    76,     1,     3,    -1,    41,     1,     3,
      -1,    44,     3,    -1,    44,   210,     3,    -1,    44,     1,
       3,    -1,     6,    -1,   210,    78,     6,    -1,    45,   212,
       3,    -1,    45,   212,    33,     6,     3,    -1,    45,   212,
      33,     7,     3,    -1,    45,   212,    33,     6,    76,     6,
       3,    -1,    45,   212,    33,     7,    76,     6,     3,    -1,
      45,   212,     1,     3,    -1,    45,    33,     6,     3,    -1,
      45,    33,     7,     3,    -1,    45,    33,     6,    76,     6,
       3,    -1,    45,    33,     7,    76,     6,     3,    -1,    45,
       1,     3,    -1,     6,    -1,   212,    78,     6,    -1,    46,
     214,     3,    -1,    46,     1,     3,    -1,   215,    -1,   214,
      78,   215,    -1,     6,    76,     6,    -1,     6,    76,     7,
      -1,     6,    76,   119,    -1,   217,   220,     9,     3,    -1,
     218,   219,     3,    -1,    42,     3,    -1,    42,     1,     3,
      -1,    42,    47,    -1,    42,     1,    47,    -1,    -1,     6,
      -1,   219,    78,     6,    -1,   219,     3,    -1,   220,   219,
       3,    -1,     1,     3,    -1,    -1,    32,     6,   222,   223,
     232,   237,     9,     3,    -1,   224,   226,     3,    -1,     1,
       3,    -1,    -1,    59,   199,    58,    -1,    -1,    59,   199,
       1,   225,    58,    -1,    -1,    33,   227,    -1,   228,    -1,
     227,    78,   228,    -1,     6,   229,    -1,    -1,    59,   230,
      58,    -1,   231,    -1,   230,    78,   231,    -1,   254,    -1,
       6,    -1,    27,    -1,    -1,   232,   233,    -1,     3,    -1,
     193,    -1,   236,    -1,   234,    -1,    -1,    38,     3,   235,
     201,   139,     9,     3,    -1,    49,     6,    76,   257,     3,
      -1,     6,    76,   257,     3,    -1,    -1,    93,   238,     3,
      -1,    93,     1,     3,    -1,     6,    -1,    82,     6,    -1,
     238,    78,     6,    -1,   238,    78,    82,     6,    -1,    -1,
      54,     6,   240,     3,   241,     9,     3,    -1,    -1,   241,
     242,    -1,     3,    -1,     6,    76,   254,   243,    -1,     6,
     243,    -1,     3,    -1,    78,    -1,    -1,    34,     6,   245,
     246,   247,   237,     9,     3,    -1,   226,     3,    -1,     1,
       3,    -1,    -1,   247,   248,    -1,     3,    -1,   193,    -1,
     236,    -1,   234,    -1,    -1,    36,   250,   251,     3,    -1,
     252,    -1,   251,    78,   252,    -1,   251,    78,     1,    -1,
       6,    -1,    35,     3,    -1,    35,   257,     3,    -1,    35,
       1,     3,    -1,     8,    -1,    55,    -1,    56,    -1,     4,
      -1,     5,    -1,     7,    -1,     6,    -1,   255,    -1,    27,
      -1,    26,    -1,   254,    -1,   256,    -1,   109,     6,    -1,
     109,    27,    -1,    99,   257,    -1,     6,    64,   257,    -1,
     257,   100,   257,    -1,   257,    99,   257,    -1,   257,   103,
     257,    -1,   257,   102,   257,    -1,   257,   101,   257,    -1,
     257,   104,   257,    -1,   257,    98,   257,    -1,   257,    97,
     257,    -1,   257,    96,   257,    -1,   257,   106,   257,    -1,
     257,   105,   257,    -1,   107,   257,    -1,   257,    87,   257,
      -1,   257,   112,    -1,   112,   257,    -1,   257,   111,    -1,
     111,   257,    -1,   257,    88,   257,    -1,   257,    86,   257,
      -1,   257,    85,   257,    -1,   257,    84,   257,    -1,   257,
      83,   257,    -1,   257,    81,   257,    -1,   257,    80,   257,
      -1,    82,   257,    -1,   257,    93,   257,    -1,   257,    92,
     257,    -1,   257,    91,   257,    -1,   257,    90,   257,    -1,
     257,    89,     6,    -1,   113,   255,    -1,   113,     4,    -1,
      95,   257,    -1,    94,   257,    -1,   266,    -1,   261,    -1,
     264,    -1,   259,    -1,   269,    -1,   271,    -1,   257,   258,
      -1,   270,    -1,   257,    61,   257,    60,    -1,   257,    61,
     103,   257,    60,    -1,   257,    62,     6,    -1,   272,    -1,
     258,    -1,   257,    76,   257,    -1,   257,    76,   257,    78,
     273,    -1,   257,    75,   257,    -1,   257,    74,   257,    -1,
     257,    73,   257,    -1,   257,    72,   257,    -1,   257,    71,
     257,    -1,   257,    65,   257,    -1,   257,    70,   257,    -1,
     257,    69,   257,    -1,   257,    68,   257,    -1,   257,    66,
     257,    -1,   257,    67,   257,    -1,    59,   257,    58,    -1,
      61,    47,    60,    -1,    61,   257,    47,    60,    -1,    61,
      47,   257,    60,    -1,    61,   257,    47,   257,    60,    -1,
      61,   257,    47,   257,    47,   257,    60,    -1,   257,    59,
     273,    58,    -1,   257,    59,    58,    -1,    -1,   257,    59,
     273,     1,   260,    58,    -1,    -1,    48,   262,   263,   201,
     139,     9,    -1,    59,   199,    58,     3,    -1,    59,   199,
       1,    -1,     1,     3,    -1,    -1,    50,   265,   263,   201,
     139,     9,    -1,    -1,    37,   267,   268,    63,   257,    -1,
     199,    -1,     1,     3,    -1,   257,    79,   257,    47,   257,
      -1,   257,    79,   257,    47,     1,    -1,   257,    79,   257,
       1,    -1,   257,    79,     1,    -1,    61,    60,    -1,    61,
     273,    60,    -1,    61,   273,     1,    -1,    52,    60,    -1,
      52,   274,    60,    -1,    52,   274,     1,    -1,    61,    63,
      60,    -1,    61,   277,    60,    -1,    61,   277,     1,    60,
      -1,   257,    -1,   273,    78,   257,    -1,   257,    -1,   274,
     275,   257,    -1,    -1,    78,    -1,   255,    -1,   276,    78,
     255,    -1,   257,    63,   257,    -1,   277,    78,   257,    63,
     257,    -1
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
    1459,  1466,  1474,  1479,  1487,  1492,  1497,  1502,  1507,  1512,
    1526,  1531,  1536,  1541,  1546,  1554,  1560,  1572,  1577,  1585,
    1586,  1590,  1594,  1598,  1610,  1617,  1627,  1628,  1631,  1632,
    1635,  1637,  1641,  1648,  1649,  1650,  1662,  1661,  1720,  1723,
    1729,  1731,  1732,  1732,  1738,  1740,  1744,  1745,  1749,  1783,
    1785,  1794,  1795,  1799,  1800,  1809,  1812,  1814,  1818,  1819,
    1822,  1840,  1844,  1844,  1878,  1900,  1927,  1929,  1930,  1937,
    1945,  1951,  1957,  1971,  1970,  2014,  2016,  2020,  2021,  2026,
    2033,  2033,  2042,  2041,  2107,  2108,  2114,  2116,  2120,  2121,
    2124,  2143,  2152,  2151,  2169,  2170,  2171,  2178,  2194,  2195,
    2196,  2206,  2207,  2208,  2209,  2210,  2211,  2215,  2233,  2234,
    2235,  2246,  2247,  2248,  2249,  2250,  2251,  2252,  2253,  2254,
    2255,  2256,  2257,  2258,  2259,  2260,  2261,  2262,  2263,  2264,
    2265,  2266,  2267,  2268,  2269,  2270,  2271,  2272,  2273,  2274,
    2275,  2276,  2277,  2278,  2279,  2280,  2281,  2282,  2283,  2284,
    2285,  2286,  2287,  2288,  2289,  2290,  2291,  2293,  2298,  2302,
    2307,  2313,  2322,  2323,  2325,  2330,  2337,  2338,  2339,  2340,
    2341,  2342,  2343,  2344,  2345,  2346,  2347,  2348,  2353,  2356,
    2359,  2362,  2365,  2371,  2377,  2382,  2382,  2392,  2391,  2432,
    2433,  2437,  2445,  2444,  2490,  2489,  2532,  2533,  2542,  2547,
    2554,  2561,  2571,  2572,  2576,  2584,  2585,  2589,  2598,  2599,
    2600,  2608,  2609,  2613,  2614,  2617,  2618,  2621,  2627,  2634,
    2635
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
  "ASSIGN_DIV", "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_TO",
  "COMMA", "QUESTION", "OR", "AND", "NOT", "LE", "GE", "LT", "GT", "NEQ",
  "EEQ", "PROVIDES", "OP_NOTIN", "OP_IN", "HASNT", "HAS", "DIESIS",
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
     365,   366,   367,   368
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   114,   115,   116,   116,   117,   117,   117,   118,   118,
     118,   118,   118,   118,   118,   118,   118,   118,   118,   119,
     119,   120,   120,   120,   121,   121,   121,   121,   121,   121,
     122,   122,   122,   122,   122,   122,   122,   122,   122,   122,
     122,   122,   122,   122,   122,   122,   122,   122,   122,   122,
     122,   123,   123,   124,   124,   126,   125,   125,   127,   127,
     127,   128,   128,   128,   130,   129,   129,   131,   131,   132,
     132,   133,   133,   134,   133,   135,   135,   137,   136,   138,
     138,   139,   139,   140,   140,   141,   141,   141,   143,   142,
     144,   142,   142,   142,   145,   145,   146,   146,   146,   146,
     147,   147,   148,   148,   148,   148,   148,   148,   149,   151,
     150,   150,   150,   153,   152,   152,   152,   155,   154,   154,
     154,   157,   156,   158,   158,   159,   159,   159,   160,   161,
     160,   162,   160,   163,   160,   164,   160,   165,   166,   165,
     167,   167,   167,   168,   168,   169,   169,   170,   170,   170,
     170,   170,   172,   171,   173,   173,   174,   174,   174,   175,
     176,   175,   177,   175,   178,   175,   179,   175,   180,   180,
     181,   181,   181,   182,   182,   182,   183,   184,   183,   185,
     185,   186,   186,   187,   187,   188,   189,   189,   189,   189,
     189,   190,   190,   191,   191,   192,   192,   193,   193,   194,
     195,   194,   194,   196,   197,   196,   198,   199,   199,   199,
     200,   201,   202,   201,   203,   201,   204,   204,   205,   205,
     206,   206,   207,   207,   207,   207,   208,   208,   208,   209,
     209,   209,   210,   210,   211,   211,   211,   211,   211,   211,
     211,   211,   211,   211,   211,   212,   212,   213,   213,   214,
     214,   215,   215,   215,   216,   216,   217,   217,   218,   218,
     219,   219,   219,   220,   220,   220,   222,   221,   223,   223,
     224,   224,   225,   224,   226,   226,   227,   227,   228,   229,
     229,   230,   230,   231,   231,   231,   232,   232,   233,   233,
     233,   233,   235,   234,   236,   236,   237,   237,   237,   238,
     238,   238,   238,   240,   239,   241,   241,   242,   242,   242,
     243,   243,   245,   244,   246,   246,   247,   247,   248,   248,
     248,   248,   250,   249,   251,   251,   251,   252,   253,   253,
     253,   254,   254,   254,   254,   254,   254,   255,   256,   256,
     256,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   258,   258,
     258,   258,   258,   259,   259,   260,   259,   262,   261,   263,
     263,   263,   265,   264,   267,   266,   268,   268,   269,   269,
     269,   269,   270,   270,   270,   271,   271,   271,   272,   272,
     272,   273,   273,   274,   274,   275,   275,   276,   276,   277,
     277
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
       3,     3,     1,     3,     3,     5,     5,     7,     7,     4,
       4,     4,     6,     6,     3,     1,     3,     3,     3,     1,
       3,     3,     3,     3,     4,     3,     2,     3,     2,     3,
       0,     1,     3,     2,     3,     2,     0,     8,     3,     2,
       0,     3,     0,     5,     0,     2,     1,     3,     2,     0,
       3,     1,     3,     1,     1,     1,     0,     2,     1,     1,
       1,     1,     0,     7,     5,     4,     0,     3,     3,     1,
       2,     3,     4,     0,     7,     0,     2,     1,     4,     2,
       1,     1,     0,     8,     2,     2,     0,     2,     1,     1,
       1,     1,     0,     4,     1,     3,     3,     1,     2,     3,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     2,     2,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     2,     3,
       2,     2,     2,     2,     3,     3,     3,     3,     3,     3,
       3,     2,     3,     3,     3,     3,     3,     2,     2,     2,
       2,     1,     1,     1,     1,     1,     1,     2,     1,     4,
       5,     3,     1,     1,     3,     5,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     4,
       4,     5,     7,     4,     3,     0,     6,     0,     6,     4,
       3,     2,     0,     6,     0,     5,     1,     2,     5,     5,
       4,     3,     2,     3,     3,     2,     3,     3,     3,     3,
       4,     1,     3,     1,     3,     0,     1,     1,     3,     3,
       5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    25,   334,   335,   337,   336,
     331,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   340,   339,     0,     0,     0,     0,     0,     0,   322,
     424,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     422,     0,     0,    59,     0,   332,   333,   108,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     4,     5,     8,    14,    24,    33,    34,    55,     0,
      38,    64,     0,    39,    40,    35,    48,    49,    50,    36,
     121,    37,   152,    41,    42,   177,    43,    10,   211,     0,
       0,    46,    47,    15,    16,    17,     9,    18,     0,   260,
      11,    13,    12,    45,    44,   341,   338,   342,   441,   393,
     384,   382,   383,   381,   385,   388,   386,   392,     0,    29,
       0,     6,     0,   337,     0,     0,     0,   417,     0,     0,
      83,     0,    85,     0,     0,     0,     0,   447,     0,     0,
       0,     0,     0,     0,     0,   441,     0,     0,   179,     0,
       0,     0,     0,   266,     0,   312,     0,   328,     0,     0,
       0,     0,     0,     0,     0,     0,   384,     0,     0,     0,
     256,   258,     0,     0,     0,   229,   232,     0,     0,   245,
       0,     0,     0,     0,     0,   249,     0,   206,     0,     0,
       0,     0,   435,   443,     0,    62,   303,     0,     0,   432,
       0,   441,     0,     0,   371,     0,   105,     0,   380,   379,
     345,     0,   103,     0,   358,   343,   344,   363,   361,   378,
     377,    81,     0,     0,     0,    57,    81,    66,   125,   156,
      81,     0,    81,   212,   214,   198,     0,     0,     0,   261,
       0,   260,     0,    30,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   362,   360,   387,     0,     0,   346,
      54,    53,     0,     0,    60,    63,    58,    61,    84,    87,
      86,    68,    70,    67,    69,    93,     0,     0,   124,   123,
       7,   155,   154,   175,     0,     0,   180,   176,   196,   195,
      28,     0,    27,     0,   330,   329,   327,     0,   324,     0,
     210,   426,   208,     0,    23,    21,    22,   221,   220,   228,
       0,   257,   259,   225,   222,     0,   231,   230,     0,   244,
       0,     0,     0,   234,     0,     0,   248,     0,   247,     0,
      26,     0,   207,   211,   211,   101,   100,   437,   436,   446,
       0,     0,   407,   408,     0,   438,     0,     0,   434,   433,
       0,   439,     0,   107,   104,   106,   102,     0,     0,     0,
       0,     0,     0,   216,   218,     0,    81,     0,   202,   204,
       0,   265,   263,     0,     0,     0,   255,   414,     0,     0,
       0,   391,   401,   405,   406,   404,   403,   402,   400,   399,
     398,   397,   396,   394,   431,     0,   370,   369,   368,   367,
     366,   365,   359,   364,   376,   375,   374,   373,   372,   355,
     354,   353,   348,   347,   351,   350,   349,   352,   357,   356,
       0,   442,     0,    51,   448,     0,     0,   174,     0,     0,
     207,   286,   274,     0,     0,     0,   316,   323,     0,   427,
       0,     0,     0,     0,     0,   374,   233,   240,     0,   241,
       0,   239,     0,     0,   246,    19,   251,   252,     0,   253,
     250,   421,     0,    81,    81,   444,   305,   410,   409,     0,
     449,   440,     0,     0,    82,     0,     0,     0,    73,    72,
      77,     0,   128,     0,     0,   126,     0,   138,     0,   159,
       0,     0,   157,     0,     0,   182,   183,    81,   217,   219,
       0,     0,   215,     0,   200,     0,   262,   254,   264,   415,
     413,     0,   389,     0,   430,     0,    31,     0,     0,    92,
      88,    90,   173,   269,     0,   296,     0,   315,   279,   275,
     276,   314,   296,   326,   325,   209,   425,   227,   226,   224,
     223,     0,     0,   235,     0,   236,     0,    20,   420,     0,
       0,     0,     0,     0,   411,     0,    56,     0,    75,     0,
       0,     0,    81,    81,   127,     0,   151,   149,   147,   148,
       0,   145,   142,     0,     0,   158,     0,   171,   172,     0,
     168,     0,     0,   186,   193,   194,     0,     0,   191,     0,
     184,     0,   197,     0,     0,     0,   199,   203,     0,   390,
     395,   429,   428,     0,    52,     0,     0,   272,   271,   288,
       0,     0,     0,     0,     0,   289,   287,   291,   290,     0,
     268,     0,   278,     0,   318,   319,   321,   320,     0,   317,
     242,   243,     0,     0,   419,   418,   423,   307,     0,     0,
     306,     0,   450,    76,    80,    79,    65,     0,     0,   133,
     135,     0,   129,   131,     0,   122,    81,     0,   139,   164,
     166,   160,   162,   170,   153,   190,     0,   188,     0,     0,
     178,   213,   205,     0,   416,    32,     0,     0,     0,    96,
       0,     0,    97,    98,    99,    91,     0,     0,   292,     0,
       0,   299,     0,     0,     0,   284,   285,     0,   281,   283,
     277,     0,   237,   238,   310,     0,   311,   309,   304,   412,
      78,    81,     0,   150,    81,     0,   146,     0,   144,    81,
       0,    81,     0,   169,   187,   192,     0,   201,     0,   109,
       0,     0,   113,     0,     0,   117,     0,     0,    95,   273,
       0,   211,     0,   298,   300,   297,     0,   267,   280,     0,
     313,     0,     0,   136,     0,   132,     0,   167,     0,   163,
     189,   112,    81,   111,   116,    81,   115,   120,    81,   119,
      89,   295,    81,     0,   301,     0,   282,   308,     0,     0,
       0,     0,   294,   302,     0,     0,     0,     0,   110,   114,
     118,   293
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    61,    62,   599,    63,   504,    65,   124,
      66,    67,   221,    68,    69,    70,   226,    71,    72,   507,
     592,   508,   509,   593,   510,   387,    73,    74,    75,   635,
     636,   710,   711,    76,    77,    78,   712,   792,   713,   795,
     714,   798,    79,   228,    80,   389,   515,   744,   745,   741,
     742,   516,   604,   517,   688,   600,   601,    81,   229,    82,
     390,   522,   751,   752,   749,   750,   609,   610,    83,    84,
     230,    85,   524,   525,   526,   527,   617,   618,    86,    87,
      88,   625,    89,   533,    90,   331,   332,   232,   396,   397,
     233,   234,    91,    92,    93,    94,   177,    95,   181,    96,
     184,   185,    97,    98,    99,   240,   241,   100,   321,   461,
     462,   716,   465,   559,   560,   652,   727,   728,   555,   646,
     647,   771,   648,   649,   723,   101,   371,   582,   670,   737,
     102,   323,   466,   562,   659,   103,   159,   327,   328,   104,
     105,   106,   107,   108,   109,   110,   628,   111,   188,   363,
     112,   189,   113,   160,   333,   114,   115,   116,   117,   118,
     194,   370,   138,   203
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -391
static const yytype_int16 yypact[] =
{
    -391,    17,   770,  -391,    52,  -391,  -391,  -391,   -29,  -391,
    -391,    62,   467,  3294,   174,   334,  3360,   468,  3426,    78,
    3492,  -391,  -391,  3558,   305,  3624,   526,   538,  3096,  -391,
    -391,   464,  3690,   540,   316,  3756,   474,   163,   549,   279,
    -391,  3822,  5224,    26,   151,  -391,  -391,  -391,  5488,  5082,
    5488,  3162,  5488,  5488,  5488,  3228,  5488,   168,  5488,  5488,
     392,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  3030,
    -391,  -391,  3030,  -391,  -391,  -391,  -391,  -391,  -391,  -391,
    -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,   138,  3030,
     130,  -391,  -391,  -391,  -391,  -391,  -391,  -391,    80,   185,
    -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  4496,  -391,
    -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,   413,  -391,
    5488,  -391,   181,  -391,    28,   148,   271,  -391,  4268,   302,
    -391,   313,  -391,   317,   285,  4328,   337,  -391,    -6,   343,
    4553,   363,   387,  4601,   399,  5940,    20,   418,  -391,  3030,
     428,  4658,   455,  -391,   478,  -391,   482,  -391,  4706,   303,
      89,   522,   539,   551,   553,  5940,   557,   558,   491,   289,
    -391,  -391,   566,  4763,   571,  -391,  -391,    57,   578,  -391,
     324,    31,   579,   507,   194,  -391,   583,  -391,   170,   170,
     586,  4811,  -391,  5940,   640,  -391,  -391,  5694,  5290,  -391,
     530,  5543,    32,    81,  6132,   589,  -391,   199,   280,   280,
     292,   590,  -391,   200,   292,  -391,  -391,   292,   292,  -391,
    -391,  -391,   542,   594,   172,  -391,  -391,  -391,  -391,  -391,
    -391,   325,  -391,  -391,  -391,  -391,   593,    88,   598,  -391,
     201,   445,   205,  -391,  5356,  5153,   591,  5488,  5488,  5488,
    5488,  5488,  5488,  5488,  5488,  5488,  5488,  5488,  5488,  3888,
    5488,  5488,  5488,  5488,  5488,  5488,  5488,  5488,   596,  5488,
    5488,  5488,  5488,  5488,  5488,  5488,  5488,  5488,  5488,  5488,
    5488,  5488,  5488,  5488,  -391,  -391,  -391,  5488,  5488,  5940,
    -391,  -391,   599,  5488,  -391,  -391,  -391,  -391,  -391,  -391,
    -391,  -391,  -391,  -391,  -391,  -391,   599,  3954,  -391,  -391,
    -391,  -391,  -391,  -391,   601,  5488,  -391,  -391,  -391,  -391,
    -391,   301,  -391,   454,  -391,  -391,  -391,   206,  -391,   604,
    -391,   532,  -391,   560,  -391,  -391,  -391,  -391,  -391,  -391,
     306,  -391,  -391,  -391,  -391,  4020,  -391,  -391,   607,  -391,
     158,   218,   621,  -391,   460,   619,  -391,     8,  -391,   620,
    -391,   627,   626,   138,   138,  -391,  -391,  -391,  -391,  -391,
    5488,   632,  -391,  -391,  5742,  -391,  5422,  5488,  -391,  -391,
     576,  -391,  5488,  -391,  -391,  -391,  -391,  1787,  1448,   556,
     561,  1561,   298,  -391,  -391,  1900,  -391,  3030,  -391,  -391,
     102,  -391,  -391,   631,   635,   209,  -391,  -391,   135,  5488,
    5591,  -391,  5940,  5940,  5940,  5940,  5940,  5940,  5940,  5940,
    5940,  5940,  5940,  5988,  -391,  4208,   417,  6132,  4125,  4125,
    4125,  4125,  4125,  4125,  -391,   280,   280,   280,   280,   516,
     516,   550,   625,   625,   263,   263,   263,   288,   292,   292,
    4382,  6036,   563,  5940,  -391,   637,  4442,  -391,   213,   639,
     626,  -391,   610,   654,   652,   656,  -391,  -391,   552,  -391,
     626,  5488,   657,   660,   661,    15,  -391,  -391,   659,  -391,
     664,  -391,   223,   224,  -391,  -391,  -391,  -391,   667,  -391,
    -391,  -391,   142,  -391,  -391,  5940,  -391,  -391,  -391,  5639,
    5940,  -391,  5796,   669,  -391,   492,  4086,   665,  -391,  -391,
    -391,   670,  -391,    50,   354,  -391,   666,  -391,   673,  -391,
      98,   671,  -391,    65,   672,   648,  -391,  -391,  -391,  -391,
     676,  2013,  -391,   624,  -391,   309,  -391,  -391,  -391,  -391,
    -391,  5844,  -391,  5488,  -391,  4152,  -391,  5488,  5488,  -391,
    -391,  -391,  -391,  -391,   152,    85,   680,  -391,   634,   613,
    -391,  -391,    90,  -391,  -391,  -391,  5940,  -391,  -391,  -391,
    -391,   691,   694,  -391,   692,  -391,   696,  -391,  -391,   700,
    2126,  2239,   176,  5488,  -391,  5488,  -391,   701,  -391,   703,
    4868,   704,  -391,  -391,  -391,   312,  -391,  -391,  -391,   633,
     121,  -391,  -391,   705,   326,  -391,   340,  -391,  -391,   178,
    -391,   708,   709,  -391,  -391,  -391,   599,    72,  -391,   710,
    -391,  1674,  -391,   711,   668,   658,  -391,  -391,   662,  -391,
     641,  -391,  6084,   215,  5940,   883,  3030,  -391,  -391,  -391,
     645,   714,   717,   718,    18,  -391,  -391,  -391,  -391,   716,
    -391,   456,  -391,   652,  -391,  -391,  -391,  -391,   723,  -391,
    -391,  -391,   730,   735,  -391,  -391,  -391,  -391,   141,   738,
    -391,  5892,  5940,  -391,  -391,  -391,  -391,  2352,  1448,  -391,
    -391,    23,  -391,  -391,    55,  -391,  -391,  3030,  -391,  -391,
    -391,  -391,  -391,   520,  -391,  -391,   739,  -391,   562,   599,
    -391,  -391,  -391,   741,  -391,  -391,   405,   483,   487,  -391,
     736,   883,  -391,  -391,  -391,  -391,   688,  5488,  -391,   678,
     752,  -391,   750,   219,   754,  -391,  -391,   -33,  -391,  -391,
    -391,   756,  -391,  -391,  -391,   544,  -391,  -391,  -391,  -391,
    -391,  -391,  3030,  -391,  -391,  3030,  -391,  2465,  -391,  -391,
    3030,  -391,  3030,  -391,  -391,  -391,   759,  -391,   760,  -391,
    3030,   762,  -391,  3030,   763,  -391,  3030,   765,  -391,  -391,
    4916,   138,  5488,  -391,  -391,  -391,    47,  -391,  -391,   456,
    -391,   220,   996,  -391,  1109,  -391,  1222,  -391,  1335,  -391,
    -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,
    -391,  -391,  -391,  4973,  -391,   766,  -391,  -391,  2578,  2691,
    2804,  2917,  -391,  -391,   781,   783,   784,   786,  -391,  -391,
    -391,  -391
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -391,  -391,  -391,  -391,  -391,  -354,  -391,    -2,  -391,  -391,
    -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,    91,
    -391,  -391,  -391,  -391,  -391,  -204,  -391,  -391,  -391,  -391,
    -391,    79,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,
    -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,
    -391,   401,  -391,  -391,  -391,  -391,   110,  -391,  -391,  -391,
    -391,  -391,  -391,  -391,  -391,  -391,  -391,   115,  -391,  -391,
    -391,  -391,  -391,  -391,   278,  -391,  -391,   119,  -391,  -390,
    -391,  -391,  -391,  -391,  -391,  -232,   349,  -315,  -391,  -391,
    -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,
    -391,   469,  -391,  -391,  -391,   -90,  -391,  -391,  -391,  -391,
    -391,  -391,   368,  -391,   179,  -391,  -391,    54,  -391,  -391,
     272,  -391,   273,   274,  -391,  -391,  -391,  -391,  -391,    56,
    -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,   370,  -391,
    -330,   -10,  -391,   -12,    -3,   807,  -391,  -391,  -391,   651,
    -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,  -391,    29,
    -391,  -391,  -391,  -391
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -446
static const yytype_int16 yytable[] =
{
      64,   128,   125,   489,   135,   400,   140,   137,   143,   242,
     473,   145,   485,   151,   486,   487,   158,     3,   570,   720,
     165,   314,   388,   173,   721,   778,   391,   485,   395,   191,
     193,   291,   352,   378,   353,   120,   197,   201,   204,   145,
     208,   209,   210,   145,   214,   779,   217,   218,   493,   494,
     220,   595,   146,   804,   485,   119,   596,   597,   598,   485,
     347,   596,   597,   598,   354,   121,   612,   225,   613,   614,
     227,   615,   306,   195,   244,   697,   245,   246,   202,   141,
     207,   238,   380,  -260,   213,   307,   239,   235,   639,   399,
     329,   640,   379,   654,   330,   330,   640,   315,   288,   606,
     722,  -170,   607,   534,   608,   286,   292,   488,   289,   355,
     288,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,   488,   641,   682,   286,   284,   285,   641,   805,
     492,   236,   286,   642,   643,   348,   539,   286,   642,   643,
     286,   381,   286,   578,   734,  -170,  -207,   317,   286,   488,
     698,   405,  -207,   637,   488,   286,   616,   196,  -260,   382,
     535,   477,   286,   699,   178,   645,  -207,  -207,   683,   179,
     286,   361,   655,   186,   215,   129,  -170,   130,   644,   667,
     470,   691,   668,   644,   290,   669,   374,   231,   286,   237,
     286,   239,   531,   540,   286,   216,   180,   358,   286,   684,
     579,   286,   384,   386,   402,   286,   286,   286,   406,   467,
     638,   286,   538,   288,   286,   286,   552,   735,   705,   736,
     470,   479,   775,   734,   293,   692,   573,   575,   554,   362,
     470,  -417,   145,   410,   478,   412,   413,   414,   415,   416,
     417,   418,   419,   420,   421,   422,   423,   425,   426,   427,
     428,   429,   430,   431,   432,   433,   693,   435,   436,   437,
     438,   439,   440,   441,   442,   443,   444,   445,   446,   447,
     448,   449,   359,   408,   294,   450,   451,   288,   288,   403,
     186,   453,   452,   403,   468,   187,   286,   403,   301,   580,
     581,   288,   341,   288,   480,   456,   454,   776,   736,   574,
     576,   528,   459,   145,  -270,   298,   147,   472,   148,   326,
       6,     7,   626,     9,    10,   679,   299,   169,   295,   170,
     300,   729,   244,   621,   245,   246,   392,   743,   393,   686,
     350,   351,   302,   475,  -270,   131,   342,   132,  -417,   244,
     305,   245,   246,   689,   458,   529,   308,   244,   133,   245,
     246,   244,   149,   245,   246,   602,   627,  -141,   495,   680,
     460,    45,    46,   171,   499,   500,   310,   281,   282,   283,
     502,   286,   394,   687,   284,   285,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   690,   677,   678,
     311,   284,   285,   282,   283,   532,   219,   541,   123,   284,
     285,  -141,   313,   284,   285,   781,   758,   286,   759,   286,
     286,   286,   286,   286,   286,   286,   286,   286,   286,   286,
     286,   316,   286,   286,   286,   286,   286,   286,   286,   286,
     286,   318,   286,   286,   286,   286,   286,   286,   286,   286,
     286,   286,   286,   286,   286,   286,   286,   286,   286,   729,
     286,   239,   760,   286,   404,   463,   802,  -274,   320,   566,
       6,     7,   725,     9,    10,   161,   482,   483,   122,   136,
     162,   163,   286,   123,   123,   174,   244,   175,   245,   246,
     176,   322,   747,   726,   761,   324,   762,   464,   764,   287,
     765,   288,   286,   587,   590,   588,   286,   286,   261,   286,
     262,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,    45,    46,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   607,   334,   608,   152,   284,   285,
     763,   145,   153,   632,   766,   145,   634,   782,   286,   154,
     784,   167,   335,   152,   155,   786,   168,   788,     6,     7,
     182,     9,    10,   563,   336,   183,   337,   511,   326,   512,
     338,   339,   518,   286,   519,  -137,   614,   340,   615,   343,
    -137,   671,   630,   672,   346,   244,   633,   245,   246,   513,
     514,   349,   356,   357,   520,   514,   360,   286,   808,   365,
     375,   809,   383,   385,   810,   154,   398,   411,   811,    45,
      46,   401,   434,  -140,   457,   123,   696,   469,  -140,   244,
     470,   245,   246,   476,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   471,   481,   484,   183,   284,   285,   286,
     491,   286,   330,   709,   715,   496,   501,   536,   537,   548,
     549,   367,   553,   464,  -445,  -445,  -445,  -445,  -445,   276,
     277,   278,   279,   280,   281,   282,   283,   557,   558,   561,
     567,   284,   285,   568,   569,   571,  -445,  -445,   286,   286,
     572,   577,   586,   594,   591,   603,   605,  -445,   523,   622,
     611,   619,   624,   650,   244,   748,   245,   246,  -445,   756,
    -445,   653,  -445,   651,   660,  -445,  -445,   661,   662,  -445,
     368,  -445,   663,   664,   673,   770,   674,   676,   685,   709,
     681,   694,   695,   700,   701,   702,   703,   718,   369,   288,
     704,   717,  -445,   187,   719,   724,   278,   279,   280,   281,
     282,   283,   731,   732,  -445,  -445,   284,   285,   733,  -445,
     783,   738,   754,   785,   757,   767,   769,  -445,   787,  -445,
     789,  -445,  -445,  -445,   772,   773,   774,   777,   793,   780,
     803,   796,   790,   791,   799,   794,   797,   286,   800,   740,
      -2,     4,   813,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,   818,    16,   819,   820,    17,   821,
     768,   521,    18,    19,   746,    20,    21,    22,    23,    24,
     286,    25,    26,   620,    27,    28,    29,    30,   753,    31,
      32,    33,    34,    35,    36,    37,    38,   755,    39,   565,
      40,    41,    42,    43,    44,    45,    46,    47,   490,    48,
     556,    49,   730,   806,   656,   657,   658,   807,   564,   166,
     364,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,     0,     0,    52,    53,     0,     0,     0,    54,
       0,     0,     0,     0,     0,    55,     0,    56,     0,    57,
       0,    58,    59,    60,     4,     0,     5,     6,     7,     8,
       9,    10,   -94,    12,    13,    14,    15,     0,    16,     0,
       0,    17,   706,   707,   708,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   222,     0,   223,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
       0,   224,     0,    40,    41,    42,    43,     0,    45,    46,
      47,     0,    48,     0,    49,     0,     0,     0,     0,     0,
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
       0,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,     0,
      52,    53,     0,     0,     0,    54,     0,     0,     0,     0,
       0,    55,     0,    56,     0,    57,     0,    58,    59,    60,
       4,     0,     5,     6,     7,     8,     9,    10,  -130,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,  -130,  -130,    20,    21,    22,    23,    24,     0,
      25,   222,     0,   223,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,  -130,   224,     0,    40,
      41,    42,    43,     0,    45,    46,    47,     0,    48,     0,
      49,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,    51,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,     0,     0,    54,     0,
       0,     0,     0,     0,    55,     0,    56,     0,    57,     0,
      58,    59,    60,     4,     0,     5,     6,     7,     8,     9,
      10,  -165,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,  -165,  -165,    20,    21,    22,
      23,    24,     0,    25,   222,     0,   223,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,  -165,
     224,     0,    40,    41,    42,    43,     0,    45,    46,    47,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,     0,     0,    52,    53,     0,     0,
       0,    54,     0,     0,     0,     0,     0,    55,     0,    56,
       0,    57,     0,    58,    59,    60,     4,     0,     5,     6,
       7,     8,     9,    10,  -161,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -161,  -161,
      20,    21,    22,    23,    24,     0,    25,   222,     0,   223,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,  -161,   224,     0,    40,    41,    42,    43,     0,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,     0,
      55,     0,    56,     0,    57,     0,    58,    59,    60,     4,
       0,     5,     6,     7,     8,     9,    10,   -71,    12,    13,
      14,    15,     0,    16,   505,   506,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     222,     0,   223,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   224,     0,    40,    41,
      42,    43,     0,    45,    46,    47,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,     0,     0,
       0,     0,     0,    55,     0,    56,     0,    57,     0,    58,
      59,    60,     4,     0,     5,     6,     7,     8,     9,    10,
    -181,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,   523,    25,   222,     0,   223,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   224,
       0,    40,    41,    42,    43,     0,    45,    46,    47,     0,
      48,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
      54,     0,     0,     0,     0,     0,    55,     0,    56,     0,
      57,     0,    58,    59,    60,     4,     0,     5,     6,     7,
       8,     9,    10,  -185,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,  -185,    25,   222,     0,   223,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   224,     0,    40,    41,    42,    43,     0,    45,
      46,    47,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,     0,    58,    59,    60,     4,     0,
       5,     6,     7,     8,     9,    10,   503,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   222,
       0,   223,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,     0,   224,     0,    40,    41,    42,
      43,     0,    45,    46,    47,     0,    48,     0,    49,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,     0,
       0,     0,    55,     0,    56,     0,    57,     0,    58,    59,
      60,     4,     0,     5,     6,     7,     8,     9,    10,   530,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   222,     0,   223,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   224,     0,
      40,    41,    42,    43,     0,    45,    46,    47,     0,    48,
       0,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,     0,     0,    52,    53,     0,     0,     0,    54,
       0,     0,     0,     0,     0,    55,     0,    56,     0,    57,
       0,    58,    59,    60,     4,     0,     5,     6,     7,     8,
       9,    10,   623,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   222,     0,   223,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
       0,   224,     0,    40,    41,    42,    43,     0,    45,    46,
      47,     0,    48,     0,    49,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
       0,     0,    54,     0,     0,     0,     0,     0,    55,     0,
      56,     0,    57,     0,    58,    59,    60,     4,     0,     5,
       6,     7,     8,     9,    10,   665,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,    24,     0,    25,   222,     0,
     223,    28,    29,    30,     0,     0,    32,     0,     0,    35,
       0,     0,     0,     0,   224,     0,    40,    41,    42,    43,
       0,    45,    46,    47,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,     0,
      52,    53,     0,     0,     0,    54,     0,     0,     0,     0,
       0,    55,     0,    56,     0,    57,     0,    58,    59,    60,
       4,     0,     5,     6,     7,     8,     9,    10,   666,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   222,     0,   223,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,     0,   224,     0,    40,
      41,    42,    43,     0,    45,    46,    47,     0,    48,     0,
      49,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,    51,     0,     0,     0,     0,
       0,     0,     0,    52,    53,     0,     0,     0,    54,     0,
       0,     0,     0,     0,    55,     0,    56,     0,    57,     0,
      58,    59,    60,     4,     0,     5,     6,     7,     8,     9,
      10,   -74,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,    24,     0,    25,   222,     0,   223,    28,    29,    30,
       0,     0,    32,     0,     0,    35,     0,     0,     0,     0,
     224,     0,    40,    41,    42,    43,     0,    45,    46,    47,
       0,    48,     0,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,     0,     0,    52,    53,     0,     0,
       0,    54,     0,     0,     0,     0,     0,    55,     0,    56,
       0,    57,     0,    58,    59,    60,     4,     0,     5,     6,
       7,     8,     9,    10,  -143,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   222,     0,   223,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   224,     0,    40,    41,    42,    43,     0,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,    51,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     0,     0,     0,
      55,     0,    56,     0,    57,     0,    58,    59,    60,     4,
       0,     5,     6,     7,     8,     9,    10,   814,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,    24,     0,    25,
     222,     0,   223,    28,    29,    30,     0,     0,    32,     0,
       0,    35,     0,     0,     0,     0,   224,     0,    40,    41,
      42,    43,     0,    45,    46,    47,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,     0,     0,
       0,     0,     0,    55,     0,    56,     0,    57,     0,    58,
      59,    60,     4,     0,     5,     6,     7,     8,     9,    10,
     815,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   222,     0,   223,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   224,
       0,    40,    41,    42,    43,     0,    45,    46,    47,     0,
      48,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,     0,     0,    52,    53,     0,     0,     0,
      54,     0,     0,     0,     0,     0,    55,     0,    56,     0,
      57,     0,    58,    59,    60,     4,     0,     5,     6,     7,
       8,     9,    10,   816,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,    24,     0,    25,   222,     0,   223,    28,
      29,    30,     0,     0,    32,     0,     0,    35,     0,     0,
       0,     0,   224,     0,    40,    41,    42,    43,     0,    45,
      46,    47,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,     0,    58,    59,    60,     4,     0,
       5,     6,     7,     8,     9,    10,   817,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   222,
       0,   223,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,     0,   224,     0,    40,    41,    42,
      43,     0,    45,    46,    47,     0,    48,     0,    49,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,    51,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,     0,
       0,     0,    55,     0,    56,     0,    57,     0,    58,    59,
      60,     4,     0,     5,     6,     7,     8,     9,    10,     0,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
       0,    25,   222,     0,   223,    28,    29,    30,     0,     0,
      32,     0,     0,    35,     0,     0,     0,     0,   224,     0,
      40,    41,    42,    43,     0,    45,    46,    47,     0,    48,
       0,    49,     0,     0,     0,     0,     0,   156,     0,   157,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,    55,     0,    56,     0,    57,
       0,    58,    59,    60,   127,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,     0,   205,     0,   206,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
      52,    53,     0,     0,     0,    54,     0,     0,     0,    30,
       0,     0,     0,    56,     0,    57,     0,    58,    59,    60,
     127,     0,    40,     0,    42,     0,     0,    45,    46,     0,
       0,    48,     0,    49,     0,     0,     0,     0,     0,   211,
       0,   212,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,    52,    53,     0,     0,
       0,    54,     0,     0,     0,    30,     0,     0,     0,    56,
       0,    57,     0,    58,    59,    60,   127,     0,    40,     0,
      42,     0,     0,    45,    46,     0,     0,    48,     0,    49,
       0,     0,     0,     0,     0,   126,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,    57,     0,    58,
      59,    60,   127,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
       0,   134,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,    52,    53,
       0,     0,     0,    54,     0,     0,     0,    30,     0,     0,
       0,    56,     0,    57,     0,    58,    59,    60,   127,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
       0,    49,     0,     0,     0,     0,     0,   139,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,    57,
       0,    58,    59,    60,   127,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,     0,   142,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
      52,    53,     0,     0,     0,    54,     0,     0,     0,    30,
       0,     0,     0,    56,     0,    57,     0,    58,    59,    60,
     127,     0,    40,     0,    42,     0,     0,    45,    46,     0,
       0,    48,     0,    49,     0,     0,     0,     0,     0,   144,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,    52,    53,     0,     0,
       0,    54,     0,     0,     0,    30,     0,     0,     0,    56,
       0,    57,     0,    58,    59,    60,   127,     0,    40,     0,
      42,     0,     0,    45,    46,     0,     0,    48,     0,    49,
       0,     0,     0,     0,     0,   150,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,    57,     0,    58,
      59,    60,   127,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
       0,   164,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,    52,    53,
       0,     0,     0,    54,     0,     0,     0,    30,     0,     0,
       0,    56,     0,    57,     0,    58,    59,    60,   127,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
       0,    49,     0,     0,     0,     0,     0,   172,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,    57,
       0,    58,    59,    60,   127,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,     0,   190,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
      52,    53,     0,     0,     0,    54,     0,     0,     0,    30,
       0,     0,     0,    56,     0,    57,     0,    58,    59,    60,
     127,     0,    40,     0,    42,     0,     0,    45,    46,     0,
       0,    48,     0,    49,     0,     0,     0,     0,     0,   424,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,    52,    53,     0,     0,
       0,    54,     0,     0,     0,    30,     0,     0,     0,    56,
       0,    57,     0,    58,    59,    60,   127,     0,    40,     0,
      42,     0,     0,    45,    46,     0,     0,    48,     0,    49,
       0,     0,     0,     0,     0,   455,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,    57,     0,    58,
      59,    60,   127,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
       0,   474,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,    52,    53,
       0,     0,     0,    54,     0,     0,     0,    30,     0,     0,
       0,    56,     0,    57,     0,    58,    59,    60,   127,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
       0,    49,     0,     0,     0,     0,     0,   589,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,    57,
       0,    58,    59,    60,   127,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,     0,   631,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
      52,    53,     0,     0,   244,    54,   245,   246,     0,    30,
       0,     0,     0,    56,     0,    57,     0,    58,    59,    60,
     127,     0,    40,     0,    42,     0,     0,    45,    46,   544,
       0,    48,     0,    49,   268,   269,   270,   271,   272,     0,
       0,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,     0,     0,    50,     0,   284,   285,     0,     0,
       0,     0,     0,     0,     0,     0,    52,    53,     0,     0,
       0,    54,     0,     0,     0,   545,     0,     0,     0,    56,
       0,    57,     0,    58,    59,    60,     0,   244,     0,   245,
     246,   296,     0,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,     0,     0,   259,   260,   261,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   297,     0,     0,     0,   284,
     285,     0,     0,     0,     0,     0,     0,   244,     0,   245,
     246,   303,     0,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,     0,     0,   259,   260,   261,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   304,     0,     0,     0,   284,
     285,     0,     0,     0,     0,   546,     0,   244,     0,   245,
     246,     0,     0,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,     0,     0,   259,   260,   261,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,     0,     0,     0,     0,   284,
     285,   244,     0,   245,   246,   550,     0,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   257,   258,     0,
     547,   259,   260,   261,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,     0,     0,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   551,
       0,     0,     0,   284,   285,     0,     0,     0,     0,   243,
       0,   244,     0,   245,   246,     0,     0,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   257,   258,     0,
       0,   259,   260,   261,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,     0,     0,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,     0,
       0,     0,     0,   284,   285,   244,   309,   245,   246,     0,
       0,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,     0,     0,   259,   260,   261,     0,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,     0,   312,     0,     0,   284,   285,     0,
       0,     0,   244,     0,   245,   246,     0,     0,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
       0,     0,   259,   260,   261,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,     0,     0,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     244,   319,   245,   246,   284,   285,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,     0,     0,
     259,   260,   261,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,     0,     0,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,     0,   325,
       0,     0,   284,   285,     0,     0,     0,   244,     0,   245,
     246,     0,     0,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,     0,     0,   259,   260,   261,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   244,   344,   245,   246,   284,
     285,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,     0,     0,   259,   260,   261,     0,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,     0,   366,     0,     0,   284,   285,     0,
       0,     0,   244,     0,   245,   246,     0,     0,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
       0,     0,   259,   260,   261,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   345,   271,   272,     0,     0,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     244,   675,   245,   246,   284,   285,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,     0,     0,
     259,   260,   261,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,     0,     0,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,     0,   801,
       0,     0,   284,   285,     0,     0,     0,   244,     0,   245,
     246,     0,     0,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,     0,     0,   259,   260,   261,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   244,   812,   245,   246,   284,
     285,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,     0,     0,   259,   260,   261,     0,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,     0,     0,     0,     0,   284,   285,     0,
       0,     0,   244,     0,   245,   246,     0,     0,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
       0,     0,   259,   260,   261,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,     0,     0,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
       0,     0,     0,     0,   284,   285,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    30,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   198,
     127,     0,    40,     0,    42,     0,     0,    45,    46,     0,
       0,    48,   199,    49,     0,   200,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     6,     7,     8,
       9,    10,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    52,    53,     0,    21,
      22,    54,     0,     0,     0,     0,     0,     0,     0,    56,
      30,    57,     0,    58,    59,    60,     0,     0,     0,     0,
     198,   127,     0,    40,     0,    42,     0,     0,    45,    46,
       0,     0,    48,     0,    49,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     6,     7,
       8,     9,    10,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    52,    53,     0,
      21,    22,    54,     0,     0,     0,   409,     0,     0,     0,
      56,    30,    57,     0,    58,    59,    60,     0,     0,     0,
       0,     0,   127,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,   192,    49,     0,     0,     0,     0,
       0,     0,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,    52,    53,
       0,     0,     0,    54,     0,     0,     0,    30,     0,     0,
       0,    56,     0,    57,     0,    58,    59,    60,   127,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
     373,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,    52,    53,     0,     0,     0,    54,
       0,     0,     0,    30,     0,     0,     0,    56,     0,    57,
       0,    58,    59,    60,   127,     0,    40,     0,    42,     0,
       0,    45,    46,     0,   407,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    21,    22,
      52,    53,     0,     0,     0,    54,     0,     0,     0,    30,
       0,     0,     0,    56,     0,    57,     0,    58,    59,    60,
     127,     0,    40,     0,    42,     0,     0,    45,    46,     0,
       0,    48,   498,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,    52,    53,     0,     0,
       0,    54,     0,     0,     0,    30,     0,     0,     0,    56,
       0,    57,     0,    58,    59,    60,   127,     0,    40,     0,
      42,     0,     0,    45,    46,     0,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,     0,     0,
     376,     0,     0,     0,     0,    56,     0,    57,     0,    58,
      59,    60,   244,     0,   245,   246,   377,     0,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
       0,     0,   259,   260,   261,     0,   262,   263,   264,   265,
     266,   267,   268,   269,   270,   271,   272,     0,   376,   273,
     274,   275,   276,   277,   278,   279,   280,   281,   282,   283,
     244,   542,   245,   246,   284,   285,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,     0,     0,
     259,   260,   261,     0,   262,   263,   264,   265,   266,   267,
     268,   269,   270,   271,   272,     0,   583,   273,   274,   275,
     276,   277,   278,   279,   280,   281,   282,   283,   244,   584,
     245,   246,   284,   285,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,     0,     0,   259,   260,
     261,     0,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,     0,     0,   273,   274,   275,   276,   277,
     278,   279,   280,   281,   282,   283,     0,     0,     0,     0,
     284,   285,   372,   244,     0,   245,   246,     0,     0,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,   257,
     258,     0,     0,   259,   260,   261,     0,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   244,   497,   245,   246,   284,   285,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   257,   258,     0,
       0,   259,   260,   261,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,     0,     0,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,     0,
       0,     0,     0,   284,   285,   244,     0,   245,   246,   585,
       0,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,   257,   258,     0,     0,   259,   260,   261,     0,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   244,   629,   245,   246,   284,   285,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,   257,
     258,     0,     0,   259,   260,   261,     0,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   244,   739,   245,   246,   284,   285,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   257,   258,     0,
       0,   259,   260,   261,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,     0,     0,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   244,
       0,   245,   246,   284,   285,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,     0,     0,   259,
     260,   261,     0,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,     0,     0,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   282,   283,   244,     0,   245,
     246,   284,   285,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   258,     0,   543,   259,   260,   261,
       0,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,     0,     0,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   283,   244,     0,   245,   246,   284,
     285,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   259,   260,   261,     0,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
       0,     0,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   244,     0,   245,   246,   284,   285,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   260,   261,     0,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,     0,     0,
     273,   274,   275,   276,   277,   278,   279,   280,   281,   282,
     283,   244,     0,   245,   246,   284,   285,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   262,   263,   264,   265,   266,
     267,   268,   269,   270,   271,   272,     0,     0,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,     0,
       0,     0,     0,   284,   285
};

static const yytype_int16 yycheck[] =
{
       2,    13,    12,   357,    16,   237,    18,    17,    20,    99,
     340,    23,     4,    25,     6,     7,    28,     0,     3,     1,
      32,     1,   226,    35,     6,    58,   230,     4,   232,    41,
      42,     3,     1,     1,     3,    64,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    78,    58,    59,   363,   364,
      60,     1,    23,     6,     4,     3,     6,     7,     8,     4,
       3,     6,     7,     8,    33,     3,     1,    69,     3,     4,
      72,     6,    78,    47,    59,     3,    61,    62,    49,     1,
      51,     1,     1,     3,    55,    91,     6,    89,     3,     1,
       1,     6,    60,     3,     6,     6,     6,    77,    78,     1,
      82,     3,     4,     1,     6,   108,    78,    99,   120,    78,
      78,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,    99,    38,     3,   128,   111,   112,    38,    82,
     362,     1,   135,    48,    49,    78,     1,   140,    48,    49,
     143,    60,   145,     1,     3,    47,    58,   149,   151,    99,
      78,   241,    63,     1,    99,   158,    91,     6,    78,    78,
      58,     3,   165,    91,     1,   555,    78,    78,    47,     6,
     173,     1,   562,     1,     6,     1,    78,     3,    93,     3,
      78,     3,     6,    93,     3,     9,   198,    49,   191,    59,
     193,     6,   396,    58,   197,    27,    33,     3,   201,    78,
      58,   204,     3,     3,     3,   208,   209,   210,     3,     3,
      58,   214,     3,    78,   217,   218,     3,    76,     3,    78,
      78,     3,     3,     3,    76,    47,     3,     3,   460,    59,
      78,    59,   244,   245,    76,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,    78,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   280,   281,
     282,   283,    78,   244,     3,   287,   288,    78,    78,    78,
       1,   293,   292,    78,    78,     6,   289,    78,     3,   493,
     494,    78,     3,    78,    76,   307,   306,    78,    78,    76,
      76,     3,     1,   315,     3,     3,     1,     1,     3,     6,
       4,     5,     3,     7,     8,     3,     3,     1,    47,     3,
       3,   651,    59,   527,    61,    62,     1,   681,     3,     3,
       6,     7,    47,   345,    33,     1,    47,     3,    59,    59,
       3,    61,    62,     3,   315,    47,     3,    59,    14,    61,
      62,    59,    47,    61,    62,     1,    47,     3,   370,    47,
      59,    55,    56,    47,   376,   377,     3,   104,   105,   106,
     382,   374,    47,    47,   111,   112,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,    47,   592,   593,
       3,   111,   112,   105,   106,   397,     4,   409,     6,   111,
     112,    47,     3,   111,   112,   735,     1,   410,     3,   412,
     413,   414,   415,   416,   417,   418,   419,   420,   421,   422,
     423,     3,   425,   426,   427,   428,   429,   430,   431,   432,
     433,     3,   435,   436,   437,   438,   439,   440,   441,   442,
     443,   444,   445,   446,   447,   448,   449,   450,   451,   779,
     453,     6,    47,   456,     9,     1,   771,     3,     3,   471,
       4,     5,     6,     7,     8,     1,     6,     7,     1,     1,
       6,     7,   475,     6,     6,     1,    59,     3,    61,    62,
       6,     3,   686,    27,     1,     3,     3,    33,     1,    76,
       3,    78,   495,     1,   506,     3,   499,   500,    81,   502,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    55,    56,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,     4,     3,     6,     1,   111,   112,
      47,   543,     6,   545,    47,   547,   548,   741,   541,     1,
     744,     1,     3,     1,     6,   749,     6,   751,     4,     5,
       1,     7,     8,     1,     3,     6,     3,     1,     6,     3,
       3,     3,     1,   566,     3,     9,     4,    76,     6,     3,
       9,   583,   543,   585,     3,    59,   547,    61,    62,    23,
      24,     3,     3,    76,    23,    24,     3,   590,   792,     3,
      60,   795,     3,     3,   798,     1,     3,     6,   802,    55,
      56,     3,     6,    47,     3,     6,   616,     3,    47,    59,
      78,    61,    62,     6,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    63,     3,     6,     6,   111,   112,   632,
       3,   634,     6,   635,   636,     3,    60,     6,     3,    76,
       3,     1,     3,    33,     4,     5,     6,     7,     8,    99,
     100,   101,   102,   103,   104,   105,   106,     3,     6,     3,
       3,   111,   112,     3,     3,     6,    26,    27,   671,   672,
       6,     4,     3,     3,     9,     9,     3,    37,    30,     3,
       9,     9,    58,     3,    59,   687,    61,    62,    48,   699,
      50,    78,    52,    59,     3,    55,    56,     3,     6,    59,
      60,    61,     6,     3,     3,   717,     3,     3,     3,   711,
      77,     3,     3,     3,     3,    47,    58,     3,    78,    78,
      58,    76,    82,     6,     6,     9,   101,   102,   103,   104,
     105,   106,     9,     3,    94,    95,   111,   112,     3,    99,
     742,     3,     3,   745,     3,     9,    58,   107,   750,   109,
     752,   111,   112,   113,    76,     3,     6,     3,   760,     3,
     772,   763,     3,     3,   766,     3,     3,   770,     3,   678,
       0,     1,     6,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,     3,    15,     3,     3,    18,     3,
     711,   390,    22,    23,   684,    25,    26,    27,    28,    29,
     803,    31,    32,   525,    34,    35,    36,    37,   693,    39,
      40,    41,    42,    43,    44,    45,    46,   698,    48,   470,
      50,    51,    52,    53,    54,    55,    56,    57,   359,    59,
     462,    61,   653,   779,   562,   562,   562,   781,   468,    32,
     189,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,
      -1,   111,   112,   113,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    19,    20,    21,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      -1,    48,    -1,    50,    51,    52,    53,    -1,    55,    56,
      57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,
      -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,
     107,    -1,   109,    -1,   111,   112,   113,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,
      24,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    47,    48,    -1,    50,    51,    52,    53,
      -1,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,
      -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,
      -1,   105,    -1,   107,    -1,   109,    -1,   111,   112,   113,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    48,    -1,    50,
      51,    52,    53,    -1,    55,    56,    57,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,
      -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,    -1,
     111,   112,   113,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    24,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    47,
      48,    -1,    50,    51,    52,    53,    -1,    55,    56,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,
      -1,   109,    -1,   111,   112,   113,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    47,    48,    -1,    50,    51,    52,    53,    -1,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,
      95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,    -1,   109,    -1,   111,   112,   113,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    16,    17,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    -1,    55,    56,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,
      -1,    -1,    -1,   105,    -1,   107,    -1,   109,    -1,   111,
     112,   113,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    30,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    -1,    55,    56,    57,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,
      99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,
     109,    -1,   111,   112,   113,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    30,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    -1,    55,
      56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,
      86,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,
      -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,
      -1,   107,    -1,   109,    -1,   111,   112,   113,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,
      53,    -1,    55,    56,    57,    -1,    59,    -1,    61,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,
      -1,    -1,   105,    -1,   107,    -1,   109,    -1,   111,   112,
     113,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    -1,    55,    56,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,
      -1,   111,   112,   113,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      -1,    48,    -1,    50,    51,    52,    53,    -1,    55,    56,
      57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,
      -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,
     107,    -1,   109,    -1,   111,   112,   113,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    -1,    31,    32,    -1,
      34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,
      -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,
      -1,    55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,
      -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,
      -1,   105,    -1,   107,    -1,   109,    -1,   111,   112,   113,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,
      51,    52,    53,    -1,    55,    56,    57,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,
      -1,    -1,    -1,    -1,   105,    -1,   107,    -1,   109,    -1,
     111,   112,   113,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    -1,    31,    32,    -1,    34,    35,    36,    37,
      -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,
      48,    -1,    50,    51,    52,    53,    -1,    55,    56,    57,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,
      -1,   109,    -1,   111,   112,   113,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    -1,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,
      95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,
     105,    -1,   107,    -1,   109,    -1,   111,   112,   113,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    29,    -1,    31,
      32,    -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,
      -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,
      52,    53,    -1,    55,    56,    57,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      82,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,
      -1,    -1,    -1,   105,    -1,   107,    -1,   109,    -1,   111,
     112,   113,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    -1,    55,    56,    57,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,    -1,
      99,    -1,    -1,    -1,    -1,    -1,   105,    -1,   107,    -1,
     109,    -1,   111,   112,   113,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    29,    -1,    31,    32,    -1,    34,    35,
      36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,
      -1,    -1,    48,    -1,    50,    51,    52,    53,    -1,    55,
      56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,
      86,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,
      -1,    -1,    -1,    99,    -1,    -1,    -1,    -1,    -1,   105,
      -1,   107,    -1,   109,    -1,   111,   112,   113,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,
      53,    -1,    55,    56,    57,    -1,    59,    -1,    61,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,
      -1,    -1,   105,    -1,   107,    -1,   109,    -1,   111,   112,
     113,     1,    -1,     3,     4,     5,     6,     7,     8,    -1,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      -1,    31,    32,    -1,    34,    35,    36,    37,    -1,    -1,
      40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,
      50,    51,    52,    53,    -1,    55,    56,    57,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,     1,    -1,     3,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    26,    27,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    37,    -1,   105,    -1,   107,    -1,   109,
      -1,   111,   112,   113,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,     1,    -1,     3,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    37,
      -1,    -1,    -1,   107,    -1,   109,    -1,   111,   112,   113,
      48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,     1,
      -1,     3,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    37,    -1,    -1,    -1,   107,
      -1,   109,    -1,   111,   112,   113,    48,    -1,    50,    -1,
      52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   107,    -1,   109,    -1,   111,
     112,   113,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    94,    95,
      -1,    -1,    -1,    99,    -1,    -1,    -1,    37,    -1,    -1,
      -1,   107,    -1,   109,    -1,   111,   112,   113,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   107,    -1,   109,
      -1,   111,   112,   113,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    37,
      -1,    -1,    -1,   107,    -1,   109,    -1,   111,   112,   113,
      48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    37,    -1,    -1,    -1,   107,
      -1,   109,    -1,   111,   112,   113,    48,    -1,    50,    -1,
      52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   107,    -1,   109,    -1,   111,
     112,   113,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    94,    95,
      -1,    -1,    -1,    99,    -1,    -1,    -1,    37,    -1,    -1,
      -1,   107,    -1,   109,    -1,   111,   112,   113,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   107,    -1,   109,
      -1,   111,   112,   113,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    37,
      -1,    -1,    -1,   107,    -1,   109,    -1,   111,   112,   113,
      48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,
      -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    37,    -1,    -1,    -1,   107,
      -1,   109,    -1,   111,   112,   113,    48,    -1,    50,    -1,
      52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   107,    -1,   109,    -1,   111,
     112,   113,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    94,    95,
      -1,    -1,    -1,    99,    -1,    -1,    -1,    37,    -1,    -1,
      -1,   107,    -1,   109,    -1,   111,   112,   113,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      -1,    61,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   107,    -1,   109,
      -1,   111,   112,   113,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      94,    95,    -1,    -1,    59,    99,    61,    62,    -1,    37,
      -1,    -1,    -1,   107,    -1,   109,    -1,   111,   112,   113,
      48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,     1,
      -1,    59,    -1,    61,    89,    90,    91,    92,    93,    -1,
      -1,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,    -1,    -1,    82,    -1,   111,   112,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    47,    -1,    -1,    -1,   107,
      -1,   109,    -1,   111,   112,   113,    -1,    59,    -1,    61,
      62,     3,    -1,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    47,    -1,    -1,    -1,   111,
     112,    -1,    -1,    -1,    -1,    -1,    -1,    59,    -1,    61,
      62,     3,    -1,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    47,    -1,    -1,    -1,   111,
     112,    -1,    -1,    -1,    -1,     3,    -1,    59,    -1,    61,
      62,    -1,    -1,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    -1,    -1,    -1,    -1,   111,
     112,    59,    -1,    61,    62,     3,    -1,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    -1,
      78,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    47,
      -1,    -1,    -1,   111,   112,    -1,    -1,    -1,    -1,     3,
      -1,    59,    -1,    61,    62,    -1,    -1,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,    -1,    -1,   111,   112,    59,     3,    61,    62,    -1,
      -1,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      -1,    -1,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    -1,     3,    -1,    -1,   111,   112,    -1,
      -1,    -1,    59,    -1,    61,    62,    -1,    -1,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      -1,    -1,    79,    80,    81,    -1,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      59,     3,    61,    62,   111,   112,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    -1,    -1,
      79,    80,    81,    -1,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    -1,    -1,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,    -1,     3,
      -1,    -1,   111,   112,    -1,    -1,    -1,    59,    -1,    61,
      62,    -1,    -1,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    59,     3,    61,    62,   111,
     112,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      -1,    -1,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    -1,     3,    -1,    -1,   111,   112,    -1,
      -1,    -1,    59,    -1,    61,    62,    -1,    -1,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      -1,    -1,    79,    80,    81,    -1,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      59,     3,    61,    62,   111,   112,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    -1,    -1,
      79,    80,    81,    -1,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    -1,    -1,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,    -1,     3,
      -1,    -1,   111,   112,    -1,    -1,    -1,    59,    -1,    61,
      62,    -1,    -1,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    59,     3,    61,    62,   111,
     112,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      -1,    -1,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    -1,    -1,    -1,    -1,   111,   112,    -1,
      -1,    -1,    59,    -1,    61,    62,    -1,    -1,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      -1,    -1,    79,    80,    81,    -1,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    -1,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      -1,    -1,    -1,    -1,   111,   112,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    47,
      48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,
      -1,    59,    60,    61,    -1,    63,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,    26,
      27,    99,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   107,
      37,   109,    -1,   111,   112,   113,    -1,    -1,    -1,    -1,
      47,    48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,
      -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    94,    95,    -1,
      26,    27,    99,    -1,    -1,    -1,   103,    -1,    -1,    -1,
     107,    37,   109,    -1,   111,   112,   113,    -1,    -1,    -1,
      -1,    -1,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    60,    61,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    94,    95,
      -1,    -1,    -1,    99,    -1,    -1,    -1,    37,    -1,    -1,
      -1,   107,    -1,   109,    -1,   111,   112,   113,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      60,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    94,    95,    -1,    -1,    -1,    99,
      -1,    -1,    -1,    37,    -1,    -1,    -1,   107,    -1,   109,
      -1,   111,   112,   113,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    58,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      94,    95,    -1,    -1,    -1,    99,    -1,    -1,    -1,    37,
      -1,    -1,    -1,   107,    -1,   109,    -1,   111,   112,   113,
      48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,
      -1,    59,    60,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    94,    95,    -1,    -1,
      -1,    99,    -1,    -1,    -1,    37,    -1,    -1,    -1,   107,
      -1,   109,    -1,   111,   112,   113,    48,    -1,    50,    -1,
      52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    94,    95,    -1,    -1,    -1,    99,    -1,    -1,
      47,    -1,    -1,    -1,    -1,   107,    -1,   109,    -1,   111,
     112,   113,    59,    -1,    61,    62,    63,    -1,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      -1,    -1,    79,    80,    81,    -1,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    -1,    47,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   106,
      59,    60,    61,    62,   111,   112,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    -1,    -1,
      79,    80,    81,    -1,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    -1,    47,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,    59,    60,
      61,    62,   111,   112,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    -1,    -1,    79,    80,
      81,    -1,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    -1,    -1,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,   106,    -1,    -1,    -1,    -1,
     111,   112,    58,    59,    -1,    61,    62,    -1,    -1,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    -1,    -1,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,    59,    60,    61,    62,   111,   112,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,    -1,    -1,   111,   112,    59,    -1,    61,    62,    63,
      -1,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      -1,    -1,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    59,    60,    61,    62,   111,   112,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    -1,    -1,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,    59,    60,    61,    62,   111,   112,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    59,
      -1,    61,    62,   111,   112,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    93,    -1,    -1,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,   106,    59,    -1,    61,
      62,   111,   112,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    76,    -1,    78,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    -1,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,    59,    -1,    61,    62,   111,
     112,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      -1,    -1,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,    59,    -1,    61,    62,   111,   112,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    -1,    -1,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,    59,    -1,    61,    62,   111,   112,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    -1,    -1,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,    -1,
      -1,    -1,    -1,   111,   112
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   115,   116,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    45,    46,    48,
      50,    51,    52,    53,    54,    55,    56,    57,    59,    61,
      82,    86,    94,    95,    99,   105,   107,   109,   111,   112,
     113,   117,   118,   120,   121,   122,   124,   125,   127,   128,
     129,   131,   132,   140,   141,   142,   147,   148,   149,   156,
     158,   171,   173,   182,   183,   185,   192,   193,   194,   196,
     198,   206,   207,   208,   209,   211,   213,   216,   217,   218,
     221,   239,   244,   249,   253,   254,   255,   256,   257,   258,
     259,   261,   264,   266,   269,   270,   271,   272,   273,     3,
      64,     3,     1,     6,   123,   255,     1,    48,   257,     1,
       3,     1,     3,    14,     1,   257,     1,   255,   276,     1,
     257,     1,     1,   257,     1,   257,   273,     1,     3,    47,
       1,   257,     1,     6,     1,     6,     1,     3,   257,   250,
     267,     1,     6,     7,     1,   257,   259,     1,     6,     1,
       3,    47,     1,   257,     1,     3,     6,   210,     1,     6,
      33,   212,     1,     6,   214,   215,     1,     6,   262,   265,
       1,   257,    60,   257,   274,    47,     6,   257,    47,    60,
      63,   257,   273,   277,   257,     1,     3,   273,   257,   257,
     257,     1,     3,   273,   257,     6,    27,   257,   257,     4,
     255,   126,    32,    34,    48,   121,   130,   121,   157,   172,
     184,    49,   201,   204,   205,   121,     1,    59,     1,     6,
     219,   220,   219,     3,    59,    61,    62,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    79,
      80,    81,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    93,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   111,   112,   258,    76,    78,   257,
       3,     3,    78,    76,     3,    47,     3,    47,     3,     3,
       3,     3,    47,     3,    47,     3,    78,    91,     3,     3,
       3,     3,     3,     3,     1,    77,     3,   121,     3,     3,
       3,   222,     3,   245,     3,     3,     6,   251,   252,     1,
       6,   199,   200,   268,     3,     3,     3,     3,     3,     3,
      76,     3,    47,     3,     3,    91,     3,     3,    78,     3,
       6,     7,     1,     3,    33,    78,     3,    76,     3,    78,
       3,     1,    59,   263,   263,     3,     3,     1,    60,    78,
     275,   240,    58,    60,   257,    60,    47,    63,     1,    60,
       1,    60,    78,     3,     3,     3,     3,   139,   139,   159,
     174,   139,     1,     3,    47,   139,   202,   203,     3,     1,
     199,     3,     3,    78,     9,   219,     3,    58,   273,   103,
     257,     6,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,     1,   257,   257,   257,   257,   257,
     257,   257,   257,   257,     6,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   255,   257,   255,     1,   257,     3,   273,     1,
      59,   223,   224,     1,    33,   226,   246,     3,    78,     3,
      78,    63,     1,   254,     1,   257,     6,     3,    76,     3,
      76,     3,     6,     7,     6,     4,     6,     7,    99,   119,
     215,     3,   199,   201,   201,   257,     3,    60,    60,   257,
     257,    60,   257,     9,   121,    16,    17,   133,   135,   136,
     138,     1,     3,    23,    24,   160,   165,   167,     1,     3,
      23,   165,   175,    30,   186,   187,   188,   189,     3,    47,
       9,   139,   121,   197,     1,    58,     6,     3,     3,     1,
      58,   257,    60,    78,     1,    47,     3,    78,    76,     3,
       3,    47,     3,     3,   199,   232,   226,     3,     6,   227,
     228,     3,   247,     1,   252,   200,   257,     3,     3,     3,
       3,     6,     6,     3,    76,     3,    76,     4,     1,    58,
     139,   139,   241,    47,    60,    63,     3,     1,     3,     1,
     257,     9,   134,   137,     3,     1,     6,     7,     8,   119,
     169,   170,     1,     9,   166,     3,     1,     4,     6,   180,
     181,     9,     1,     3,     4,     6,    91,   190,   191,     9,
     188,   139,     3,     9,    58,   195,     3,    47,   260,    60,
     273,     1,   257,   273,   257,   143,   144,     1,    58,     3,
       6,    38,    48,    49,    93,   193,   233,   234,   236,   237,
       3,    59,   229,    78,     3,   193,   234,   236,   237,   248,
       3,     3,     6,     6,     3,     9,     9,     3,     6,     9,
     242,   257,   257,     3,     3,     3,     3,   139,   139,     3,
      47,    77,     3,    47,    78,     3,     3,    47,   168,     3,
      47,     3,    47,    78,     3,     3,   255,     3,    78,    91,
       3,     3,    47,    58,    58,     3,    19,    20,    21,   121,
     145,   146,   150,   152,   154,   121,   225,    76,     3,     6,
       1,     6,    82,   238,     9,     6,    27,   230,   231,   254,
     228,     9,     3,     3,     3,    76,    78,   243,     3,    60,
     133,   163,   164,   119,   161,   162,   170,   139,   121,   178,
     179,   176,   177,   181,     3,   191,   255,     3,     1,     3,
      47,     1,     3,    47,     1,     3,    47,     9,   145,    58,
     257,   235,    76,     3,     6,     3,    78,     3,    58,    78,
       3,   254,   139,   121,   139,   121,   139,   121,   139,   121,
       3,     3,   151,   121,     3,   153,   121,     3,   155,   121,
       3,     3,   201,   257,     6,    82,   231,   243,   139,   139,
     139,   139,     3,     6,     9,     9,     9,     9,     3,     3,
       3,     3
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
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 238:
#line 1508 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 239:
#line 1513 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 240:
#line 1527 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 241:
#line 1532 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 242:
#line 1537 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 243:
#line 1542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 244:
#line 1547 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 245:
#line 1555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *(yyvsp[(1) - (1)].stringp) ) );
         (yyval.fal_genericList) = lst;
      }
    break;

  case 246:
#line 1561 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(1) - (3)].fal_genericList)->pushBack( new Falcon::String( *(yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_genericList) = (yyvsp[(1) - (3)].fal_genericList);
      }
    break;

  case 247:
#line 1573 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 248:
#line 1578 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 251:
#line 1591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 252:
#line 1595 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 253:
#line 1599 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 254:
#line 1613 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 255:
#line 1620 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 257:
#line 1628 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); }
    break;

  case 259:
#line 1632 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); }
    break;

  case 261:
#line 1638 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         }
    break;

  case 262:
#line 1642 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         }
    break;

  case 265:
#line 1651 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   }
    break;

  case 266:
#line 1662 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 267:
#line 1696 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 269:
#line 1724 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 272:
#line 1732 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 273:
#line 1733 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 278:
#line 1750 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 279:
#line 1783 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; }
    break;

  case 280:
#line 1788 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
   }
    break;

  case 281:
#line 1794 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 282:
#line 1795 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 284:
#line 1801 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      }
    break;

  case 285:
#line 1809 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 289:
#line 1819 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 290:
#line 1822 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 292:
#line 1844 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 293:
#line 1868 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 294:
#line 1879 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 295:
#line 1901 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 298:
#line 1931 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   }
    break;

  case 299:
#line 1938 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      }
    break;

  case 300:
#line 1946 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      }
    break;

  case 301:
#line 1952 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      }
    break;

  case 302:
#line 1958 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      }
    break;

  case 303:
#line 1971 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 304:
#line 2005 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 308:
#line 2022 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 309:
#line 2027 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 312:
#line 2042 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 313:
#line 2084 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 315:
#line 2109 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 319:
#line 2121 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 320:
#line 2124 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 322:
#line 2152 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 323:
#line 2157 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 326:
#line 2172 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 327:
#line 2179 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 328:
#line 2194 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 329:
#line 2195 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 330:
#line 2196 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 331:
#line 2206 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 332:
#line 2207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 333:
#line 2208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 334:
#line 2209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 335:
#line 2210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 336:
#line 2211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 337:
#line 2216 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 339:
#line 2234 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 340:
#line 2235 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); }
    break;

  case 343:
#line 2248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( (yyvsp[(2) - (2)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 344:
#line 2249 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString("self") ); /* do not add the symbol to the compiler */ }
    break;

  case 345:
#line 2250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 346:
#line 2251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_fbind, new Falcon::Value((yyvsp[(1) - (3)].stringp)), (yyvsp[(3) - (3)].fal_val)) ); }
    break;

  case 347:
#line 2252 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 348:
#line 2253 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 349:
#line 2254 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 350:
#line 2255 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 351:
#line 2256 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 352:
#line 2257 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 353:
#line 2258 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 354:
#line 2259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 355:
#line 2260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 356:
#line 2261 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 357:
#line 2262 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 358:
#line 2263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 359:
#line 2264 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 360:
#line 2265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 361:
#line 2266 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 362:
#line 2267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 363:
#line 2268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 364:
#line 2269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 365:
#line 2270 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 366:
#line 2271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 367:
#line 2272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 368:
#line 2273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 369:
#line 2274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 370:
#line 2275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 371:
#line 2276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 372:
#line 2277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 373:
#line 2278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 374:
#line 2279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 375:
#line 2280 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 376:
#line 2281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 377:
#line 2282 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 378:
#line 2283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 379:
#line 2284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 380:
#line 2285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 387:
#line 2293 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 388:
#line 2298 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 389:
#line 2302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 390:
#line 2307 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 391:
#line 2313 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 394:
#line 2325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 395:
#line 2330 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 396:
#line 2337 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 397:
#line 2338 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 398:
#line 2339 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 399:
#line 2340 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 400:
#line 2341 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 401:
#line 2342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 402:
#line 2343 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 403:
#line 2344 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 404:
#line 2345 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 405:
#line 2346 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 406:
#line 2347 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 407:
#line 2348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 408:
#line 2353 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 409:
#line 2356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 410:
#line 2359 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 411:
#line 2362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 412:
#line 2365 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 413:
#line 2372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 414:
#line 2378 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 415:
#line 2382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 416:
#line 2383 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 417:
#line 2392 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 418:
#line 2426 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 420:
#line 2434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 421:
#line 2438 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 422:
#line 2445 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 423:
#line 2478 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 424:
#line 2490 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 425:
#line 2522 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            COMPILER->checkLocalUndefined();
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 427:
#line 2534 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      }
    break;

  case 428:
#line 2543 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 429:
#line 2548 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 430:
#line 2555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 431:
#line 2562 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 432:
#line 2571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 433:
#line 2573 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 434:
#line 2577 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 435:
#line 2584 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 436:
#line 2586 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 437:
#line 2590 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 438:
#line 2598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 439:
#line 2599 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 440:
#line 2601 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 441:
#line 2608 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 442:
#line 2609 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 443:
#line 2613 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 444:
#line 2614 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 447:
#line 2621 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 448:
#line 2627 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 449:
#line 2634 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 450:
#line 2635 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;


/* Line 1267 of yacc.c.  */
#line 6411 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2639 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


