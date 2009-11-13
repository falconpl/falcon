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
     SELF = 281,
     FSELF = 282,
     TRY = 283,
     CATCH = 284,
     RAISE = 285,
     CLASS = 286,
     FROM = 287,
     OBJECT = 288,
     RETURN = 289,
     GLOBAL = 290,
     INIT = 291,
     LOAD = 292,
     LAUNCH = 293,
     CONST_KW = 294,
     EXPORT = 295,
     IMPORT = 296,
     DIRECTIVE = 297,
     COLON = 298,
     FUNCDECL = 299,
     STATIC = 300,
     INNERFUNC = 301,
     FORDOT = 302,
     LISTPAR = 303,
     LOOP = 304,
     ENUM = 305,
     TRUE_TOKEN = 306,
     FALSE_TOKEN = 307,
     STATE = 308,
     OUTER_STRING = 309,
     CLOSEPAR = 310,
     OPENPAR = 311,
     CLOSESQUARE = 312,
     OPENSQUARE = 313,
     DOT = 314,
     OPEN_GRAPH = 315,
     CLOSE_GRAPH = 316,
     ARROW = 317,
     VBAR = 318,
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
     OP_AS = 331,
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
     DIESIS = 347,
     ATSIGN = 348,
     CAP_CAP = 349,
     VBAR_VBAR = 350,
     AMPER_AMPER = 351,
     MINUS = 352,
     PLUS = 353,
     PERCENT = 354,
     SLASH = 355,
     STAR = 356,
     POW = 357,
     SHR = 358,
     SHL = 359,
     CAP_XOROOB = 360,
     CAP_ISOOB = 361,
     CAP_DEOOB = 362,
     CAP_OOB = 363,
     CAP_EVAL = 364,
     TILDE = 365,
     NEG = 366,
     AMPER = 367,
     DECREMENT = 368,
     INCREMENT = 369,
     DOLLAR = 370
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
#define SELF 281
#define FSELF 282
#define TRY 283
#define CATCH 284
#define RAISE 285
#define CLASS 286
#define FROM 287
#define OBJECT 288
#define RETURN 289
#define GLOBAL 290
#define INIT 291
#define LOAD 292
#define LAUNCH 293
#define CONST_KW 294
#define EXPORT 295
#define IMPORT 296
#define DIRECTIVE 297
#define COLON 298
#define FUNCDECL 299
#define STATIC 300
#define INNERFUNC 301
#define FORDOT 302
#define LISTPAR 303
#define LOOP 304
#define ENUM 305
#define TRUE_TOKEN 306
#define FALSE_TOKEN 307
#define STATE 308
#define OUTER_STRING 309
#define CLOSEPAR 310
#define OPENPAR 311
#define CLOSESQUARE 312
#define OPENSQUARE 313
#define DOT 314
#define OPEN_GRAPH 315
#define CLOSE_GRAPH 316
#define ARROW 317
#define VBAR 318
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
#define OP_AS 331
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
#define DIESIS 347
#define ATSIGN 348
#define CAP_CAP 349
#define VBAR_VBAR 350
#define AMPER_AMPER 351
#define MINUS 352
#define PLUS 353
#define PERCENT 354
#define SLASH 355
#define STAR 356
#define POW 357
#define SHR 358
#define SHL 359
#define CAP_XOROOB 360
#define CAP_ISOOB 361
#define CAP_DEOOB 362
#define CAP_OOB 363
#define CAP_EVAL 364
#define TILDE 365
#define NEG 366
#define AMPER 367
#define DECREMENT 368
#define INCREMENT 369
#define DOLLAR 370




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
#define  CTX_LINE  ( COMPILER->lexer()->contextStart() )
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
   Falcon::List *fal_genericList;
}
/* Line 187 of yacc.c.  */
#line 391 "/home/user/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 404 "/home/user/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   6704

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  116
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  164
/* YYNRULES -- Number of rules.  */
#define YYNRULES  462
/* YYNRULES -- Number of states.  */
#define YYNSTATES  848

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   370

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
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      43,    47,    51,    55,    57,    59,    61,    65,    69,    73,
      76,    79,    84,    91,    93,    95,    97,    99,   101,   103,
     105,   107,   109,   111,   113,   115,   117,   119,   121,   123,
     125,   127,   131,   137,   141,   145,   146,   152,   155,   159,
     163,   167,   171,   172,   180,   184,   188,   189,   191,   192,
     199,   202,   206,   210,   214,   218,   219,   221,   222,   226,
     229,   233,   234,   239,   243,   247,   248,   251,   254,   258,
     261,   265,   269,   270,   277,   278,   285,   291,   295,   298,
     303,   308,   313,   317,   318,   321,   324,   325,   328,   330,
     332,   334,   336,   340,   344,   348,   351,   355,   358,   362,
     366,   368,   369,   376,   380,   384,   385,   392,   396,   400,
     401,   408,   412,   416,   417,   424,   428,   432,   433,   436,
     440,   442,   443,   449,   450,   456,   457,   463,   464,   470,
     471,   472,   476,   477,   479,   482,   485,   488,   490,   494,
     496,   498,   500,   504,   506,   507,   514,   518,   522,   523,
     526,   530,   532,   533,   539,   540,   546,   547,   553,   554,
     560,   562,   566,   567,   569,   571,   575,   576,   583,   586,
     590,   591,   593,   595,   598,   601,   604,   609,   613,   619,
     623,   625,   629,   631,   633,   637,   641,   647,   650,   656,
     657,   665,   669,   675,   676,   683,   686,   687,   689,   693,
     695,   696,   697,   703,   704,   708,   711,   715,   718,   722,
     726,   730,   736,   742,   746,   749,   753,   757,   759,   763,
     767,   773,   779,   787,   795,   803,   811,   816,   821,   826,
     831,   838,   845,   849,   854,   859,   861,   865,   869,   873,
     875,   879,   883,   887,   891,   892,   900,   904,   907,   908,
     912,   913,   919,   920,   923,   925,   929,   932,   933,   936,
     940,   941,   944,   946,   948,   950,   952,   954,   956,   957,
     965,   971,   976,   983,   985,   988,   989,   997,   998,  1001,
    1003,  1008,  1010,  1013,  1015,  1017,  1018,  1026,  1029,  1032,
    1033,  1036,  1038,  1040,  1042,  1044,  1046,  1047,  1052,  1054,
    1056,  1059,  1063,  1067,  1069,  1072,  1076,  1080,  1082,  1084,
    1086,  1088,  1090,  1092,  1094,  1096,  1098,  1100,  1102,  1104,
    1106,  1108,  1110,  1112,  1113,  1115,  1117,  1119,  1122,  1125,
    1128,  1132,  1136,  1140,  1143,  1147,  1152,  1157,  1162,  1167,
    1172,  1177,  1182,  1187,  1192,  1197,  1202,  1205,  1209,  1212,
    1215,  1218,  1221,  1225,  1229,  1233,  1237,  1241,  1245,  1249,
    1252,  1256,  1260,  1264,  1267,  1270,  1273,  1276,  1279,  1282,
    1285,  1288,  1291,  1293,  1295,  1297,  1299,  1301,  1303,  1306,
    1308,  1313,  1319,  1323,  1325,  1327,  1331,  1337,  1341,  1345,
    1349,  1353,  1357,  1361,  1365,  1369,  1373,  1377,  1381,  1385,
    1389,  1394,  1399,  1405,  1413,  1418,  1422,  1423,  1430,  1431,
    1438,  1439,  1446,  1451,  1455,  1458,  1461,  1464,  1467,  1468,
    1475,  1481,  1487,  1492,  1496,  1499,  1503,  1507,  1510,  1514,
    1518,  1522,  1526,  1531,  1533,  1537,  1539,  1543,  1544,  1546,
    1548,  1552,  1556
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     117,     0,    -1,   118,    -1,    -1,   118,   119,    -1,   120,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   122,    -1,
     220,    -1,   200,    -1,   223,    -1,   244,    -1,   239,    -1,
     123,    -1,   214,    -1,   215,    -1,   217,    -1,     4,    -1,
      97,     4,    -1,    37,     6,     3,    -1,    37,     7,     3,
      -1,    37,     1,     3,    -1,   124,    -1,   218,    -1,     3,
      -1,    44,     1,     3,    -1,    33,     1,     3,    -1,    31,
       1,     3,    -1,     1,     3,    -1,   259,     3,    -1,   275,
      75,   259,     3,    -1,   275,    75,   259,    78,   275,     3,
      -1,   126,    -1,   127,    -1,   131,    -1,   147,    -1,   164,
      -1,   179,    -1,   134,    -1,   145,    -1,   146,    -1,   190,
      -1,   199,    -1,   253,    -1,   249,    -1,   213,    -1,   155,
      -1,   156,    -1,   157,    -1,   256,    -1,   256,    75,   259,
      -1,   125,    78,   256,    75,   259,    -1,    10,   125,     3,
      -1,    10,     1,     3,    -1,    -1,   129,   128,   144,     9,
       3,    -1,   130,   123,    -1,    11,   259,     3,    -1,    11,
       1,     3,    -1,    11,   259,    43,    -1,    11,     1,    43,
      -1,    -1,    49,     3,   132,   144,     9,   133,     3,    -1,
      49,    43,   123,    -1,    49,     1,     3,    -1,    -1,   259,
      -1,    -1,   136,   135,   144,   138,     9,     3,    -1,   137,
     123,    -1,    15,   259,     3,    -1,    15,     1,     3,    -1,
      15,   259,    43,    -1,    15,     1,    43,    -1,    -1,   141,
      -1,    -1,   140,   139,   144,    -1,    16,     3,    -1,    16,
       1,     3,    -1,    -1,   143,   142,   144,   138,    -1,    17,
     259,     3,    -1,    17,     1,     3,    -1,    -1,   144,   123,
      -1,    12,     3,    -1,    12,     1,     3,    -1,    13,     3,
      -1,    13,    14,     3,    -1,    13,     1,     3,    -1,    -1,
      18,   278,    91,   259,   148,   150,    -1,    -1,    18,   256,
      75,   151,   149,   150,    -1,    18,   278,    91,     1,     3,
      -1,    18,     1,     3,    -1,    43,   123,    -1,     3,   153,
       9,     3,    -1,   259,    77,   259,   152,    -1,   259,    77,
     259,     1,    -1,   259,    77,     1,    -1,    -1,    78,   259,
      -1,    78,     1,    -1,    -1,   154,   153,    -1,   123,    -1,
     158,    -1,   160,    -1,   162,    -1,    47,   259,     3,    -1,
      47,     1,     3,    -1,   103,   275,     3,    -1,   103,     3,
      -1,    86,   275,     3,    -1,    86,     3,    -1,   103,     1,
       3,    -1,    86,     1,     3,    -1,    54,    -1,    -1,    19,
       3,   159,   144,     9,     3,    -1,    19,    43,   123,    -1,
      19,     1,     3,    -1,    -1,    20,     3,   161,   144,     9,
       3,    -1,    20,    43,   123,    -1,    20,     1,     3,    -1,
      -1,    21,     3,   163,   144,     9,     3,    -1,    21,    43,
     123,    -1,    21,     1,     3,    -1,    -1,   166,   165,   167,
     173,     9,     3,    -1,    22,   259,     3,    -1,    22,     1,
       3,    -1,    -1,   167,   168,    -1,   167,     1,     3,    -1,
       3,    -1,    -1,    23,   177,     3,   169,   144,    -1,    -1,
      23,   177,    43,   170,   123,    -1,    -1,    23,     1,     3,
     171,   144,    -1,    -1,    23,     1,    43,   172,   123,    -1,
      -1,    -1,   175,   174,   176,    -1,    -1,    24,    -1,    24,
       1,    -1,     3,   144,    -1,    43,   123,    -1,   178,    -1,
     177,    78,   178,    -1,     8,    -1,   121,    -1,     7,    -1,
     121,    77,   121,    -1,     6,    -1,    -1,   181,   180,   182,
     173,     9,     3,    -1,    25,   259,     3,    -1,    25,     1,
       3,    -1,    -1,   182,   183,    -1,   182,     1,     3,    -1,
       3,    -1,    -1,    23,   188,     3,   184,   144,    -1,    -1,
      23,   188,    43,   185,   123,    -1,    -1,    23,     1,     3,
     186,   144,    -1,    -1,    23,     1,    43,   187,   123,    -1,
     189,    -1,   188,    78,   189,    -1,    -1,     4,    -1,     6,
      -1,    28,    43,   123,    -1,    -1,   192,   191,   144,   193,
       9,     3,    -1,    28,     3,    -1,    28,     1,     3,    -1,
      -1,   194,    -1,   195,    -1,   194,   195,    -1,   196,   144,
      -1,    29,     3,    -1,    29,    91,   256,     3,    -1,    29,
     197,     3,    -1,    29,   197,    91,   256,     3,    -1,    29,
       1,     3,    -1,   198,    -1,   197,    78,   198,    -1,     4,
      -1,     6,    -1,    30,   259,     3,    -1,    30,     1,     3,
      -1,   201,   208,   144,     9,     3,    -1,   203,   123,    -1,
     205,    56,   206,    55,     3,    -1,    -1,   205,    56,   206,
       1,   202,    55,     3,    -1,   205,     1,     3,    -1,   205,
      56,   206,    55,    43,    -1,    -1,   205,    56,     1,   204,
      55,    43,    -1,    44,     6,    -1,    -1,   207,    -1,   206,
      78,   207,    -1,     6,    -1,    -1,    -1,   211,   209,   144,
       9,     3,    -1,    -1,   212,   210,   123,    -1,    45,     3,
      -1,    45,     1,     3,    -1,    45,    43,    -1,    45,     1,
      43,    -1,    38,   261,     3,    -1,    38,     1,     3,    -1,
      39,     6,    75,   255,     3,    -1,    39,     6,    75,     1,
       3,    -1,    39,     1,     3,    -1,    40,     3,    -1,    40,
     216,     3,    -1,    40,     1,     3,    -1,     6,    -1,   216,
      78,     6,    -1,    41,   219,     3,    -1,    41,   219,    32,
       6,     3,    -1,    41,   219,    32,     7,     3,    -1,    41,
     219,    32,     6,    76,     6,     3,    -1,    41,   219,    32,
       7,    76,     6,     3,    -1,    41,   219,    32,     6,    91,
       6,     3,    -1,    41,   219,    32,     7,    91,     6,     3,
      -1,    41,     6,     1,     3,    -1,    41,   219,     1,     3,
      -1,    41,    32,     6,     3,    -1,    41,    32,     7,     3,
      -1,    41,    32,     6,    76,     6,     3,    -1,    41,    32,
       7,    76,     6,     3,    -1,    41,     1,     3,    -1,     6,
      43,   255,     3,    -1,     6,    43,     1,     3,    -1,     6,
      -1,   219,    78,     6,    -1,    42,   221,     3,    -1,    42,
       1,     3,    -1,   222,    -1,   221,    78,   222,    -1,     6,
      75,     6,    -1,     6,    75,     7,    -1,     6,    75,   121,
      -1,    -1,    31,     6,   224,   225,   232,     9,     3,    -1,
     226,   228,     3,    -1,     1,     3,    -1,    -1,    56,   206,
      55,    -1,    -1,    56,   206,     1,   227,    55,    -1,    -1,
      32,   229,    -1,   230,    -1,   229,    78,   230,    -1,     6,
     231,    -1,    -1,    56,    55,    -1,    56,   275,    55,    -1,
      -1,   232,   233,    -1,     3,    -1,   200,    -1,   236,    -1,
     234,    -1,   218,    -1,   237,    -1,    -1,    36,     3,   235,
     208,   144,     9,     3,    -1,    45,     6,    75,   255,     3,
      -1,     6,    75,   259,     3,    -1,    53,     6,     3,   238,
       9,     3,    -1,   201,    -1,   238,   201,    -1,    -1,    50,
       6,   240,     3,   241,     9,     3,    -1,    -1,   241,   242,
      -1,     3,    -1,     6,    75,   255,   243,    -1,   218,    -1,
       6,   243,    -1,     3,    -1,    78,    -1,    -1,    33,     6,
     245,   246,   247,     9,     3,    -1,   228,     3,    -1,     1,
       3,    -1,    -1,   247,   248,    -1,     3,    -1,   200,    -1,
     236,    -1,   234,    -1,   218,    -1,    -1,    35,   250,   251,
       3,    -1,   252,    -1,     1,    -1,   252,     1,    -1,   251,
      78,   252,    -1,   251,    78,     1,    -1,     6,    -1,    34,
       3,    -1,    34,   259,     3,    -1,    34,     1,     3,    -1,
       8,    -1,    51,    -1,    52,    -1,     4,    -1,     5,    -1,
       7,    -1,     8,    -1,    51,    -1,    52,    -1,   121,    -1,
       5,    -1,     7,    -1,     6,    -1,   256,    -1,    26,    -1,
      27,    -1,    -1,     3,    -1,   254,    -1,   257,    -1,   112,
       6,    -1,   112,     4,    -1,   112,    26,    -1,   112,    59,
       6,    -1,   112,    59,     4,    -1,   112,    59,    26,    -1,
      97,   259,    -1,     6,    63,   259,    -1,   259,    98,   258,
     259,    -1,   259,    97,   258,   259,    -1,   259,   101,   258,
     259,    -1,   259,   100,   258,   259,    -1,   259,    99,   258,
     259,    -1,   259,   102,   258,   259,    -1,   259,    96,   258,
     259,    -1,   259,    95,   258,   259,    -1,   259,    94,   258,
     259,    -1,   259,   104,   258,   259,    -1,   259,   103,   258,
     259,    -1,   110,   259,    -1,   259,    87,   259,    -1,   259,
     114,    -1,   114,   259,    -1,   259,   113,    -1,   113,   259,
      -1,   259,    88,   259,    -1,   259,    86,   259,    -1,   259,
      85,   259,    -1,   259,    84,   259,    -1,   259,    83,   259,
      -1,   259,    81,   259,    -1,   259,    80,   259,    -1,    82,
     259,    -1,   259,    91,   259,    -1,   259,    90,   259,    -1,
     259,    89,     6,    -1,   115,   256,    -1,   115,     4,    -1,
      93,   259,    -1,    92,   259,    -1,   109,   259,    -1,   108,
     259,    -1,   107,   259,    -1,   106,   259,    -1,   105,   259,
      -1,   263,    -1,   265,    -1,   269,    -1,   261,    -1,   271,
      -1,   273,    -1,   259,   260,    -1,   272,    -1,   259,    58,
     259,    57,    -1,   259,    58,   101,   259,    57,    -1,   259,
      59,     6,    -1,   274,    -1,   260,    -1,   259,    75,   259,
      -1,   259,    75,   259,    78,   275,    -1,   259,    74,   259,
      -1,   259,    73,   259,    -1,   259,    72,   259,    -1,   259,
      71,   259,    -1,   259,    70,   259,    -1,   259,    64,   259,
      -1,   259,    69,   259,    -1,   259,    68,   259,    -1,   259,
      67,   259,    -1,   259,    65,   259,    -1,   259,    66,   259,
      -1,    56,   259,    55,    -1,    58,    43,    57,    -1,    58,
     259,    43,    57,    -1,    58,    43,   259,    57,    -1,    58,
     259,    43,   259,    57,    -1,    58,   259,    43,   259,    43,
     259,    57,    -1,   259,    56,   275,    55,    -1,   259,    56,
      55,    -1,    -1,   259,    56,   275,     1,   262,    55,    -1,
      -1,    44,   264,   267,   208,   144,     9,    -1,    -1,    60,
     266,   268,   208,   144,    61,    -1,    56,   206,    55,     3,
      -1,    56,   206,     1,    -1,     1,     3,    -1,   206,    62,
      -1,   206,     1,    -1,     1,    62,    -1,    -1,    46,   270,
     267,   208,   144,     9,    -1,   259,    79,   259,    43,   259,
      -1,   259,    79,   259,    43,     1,    -1,   259,    79,   259,
       1,    -1,   259,    79,     1,    -1,    58,    57,    -1,    58,
     275,    57,    -1,    58,   275,     1,    -1,    48,    57,    -1,
      48,   276,    57,    -1,    48,   276,     1,    -1,    58,    62,
      57,    -1,    58,   279,    57,    -1,    58,   279,     1,    57,
      -1,   259,    -1,   275,    78,   259,    -1,   259,    -1,   276,
     277,   259,    -1,    -1,    78,    -1,   256,    -1,   278,    78,
     256,    -1,   259,    62,   259,    -1,   279,    78,   259,    62,
     259,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   197,   197,   200,   202,   206,   207,   208,   212,   213,
     214,   219,   224,   229,   234,   239,   240,   241,   245,   246,
     250,   256,   262,   269,   270,   271,   272,   273,   274,   275,
     280,   291,   297,   311,   312,   313,   314,   315,   316,   317,
     318,   319,   320,   321,   322,   323,   324,   325,   326,   327,
     331,   334,   340,   348,   350,   355,   355,   369,   377,   378,
     382,   383,   387,   387,   402,   408,   415,   416,   420,   420,
     435,   445,   446,   450,   451,   455,   457,   458,   458,   467,
     468,   473,   473,   485,   486,   489,   491,   497,   506,   514,
     524,   533,   543,   542,   563,   562,   585,   590,   598,   604,
     611,   617,   621,   628,   629,   630,   633,   635,   639,   646,
     647,   648,   652,   665,   673,   677,   683,   689,   696,   701,
     710,   720,   720,   734,   743,   747,   747,   760,   769,   773,
     773,   789,   798,   802,   802,   819,   820,   827,   829,   830,
     834,   836,   835,   846,   846,   858,   858,   870,   870,   886,
     889,   888,   901,   902,   903,   906,   907,   913,   914,   918,
     927,   939,   950,   961,   982,   982,   999,  1000,  1007,  1009,
    1010,  1014,  1016,  1015,  1026,  1026,  1039,  1039,  1051,  1051,
    1069,  1070,  1073,  1074,  1086,  1107,  1114,  1113,  1132,  1133,
    1136,  1138,  1142,  1143,  1147,  1152,  1170,  1190,  1200,  1211,
    1219,  1220,  1224,  1236,  1259,  1260,  1267,  1277,  1286,  1287,
    1287,  1291,  1295,  1296,  1296,  1303,  1374,  1376,  1377,  1381,
    1396,  1399,  1398,  1410,  1409,  1424,  1425,  1429,  1430,  1439,
    1443,  1451,  1461,  1466,  1478,  1487,  1494,  1502,  1507,  1515,
    1520,  1525,  1530,  1550,  1569,  1574,  1579,  1584,  1598,  1603,
    1608,  1613,  1618,  1627,  1632,  1639,  1645,  1657,  1662,  1670,
    1671,  1675,  1679,  1683,  1697,  1696,  1759,  1762,  1768,  1770,
    1771,  1771,  1777,  1779,  1783,  1784,  1788,  1812,  1813,  1814,
    1821,  1823,  1827,  1828,  1831,  1849,  1850,  1851,  1855,  1855,
    1889,  1911,  1939,  1945,  1946,  1955,  1954,  1998,  2000,  2004,
    2005,  2009,  2010,  2017,  2017,  2026,  2025,  2092,  2093,  2099,
    2101,  2105,  2106,  2109,  2128,  2129,  2138,  2137,  2155,  2156,
    2161,  2166,  2167,  2174,  2190,  2191,  2192,  2200,  2201,  2202,
    2203,  2204,  2205,  2209,  2210,  2211,  2212,  2213,  2214,  2218,
    2236,  2237,  2238,  2258,  2260,  2264,  2265,  2266,  2267,  2268,
    2269,  2270,  2271,  2272,  2273,  2274,  2300,  2301,  2321,  2345,
    2362,  2363,  2364,  2365,  2366,  2367,  2368,  2369,  2370,  2371,
    2372,  2373,  2374,  2375,  2376,  2377,  2378,  2379,  2380,  2381,
    2382,  2383,  2384,  2385,  2386,  2387,  2388,  2389,  2390,  2391,
    2392,  2393,  2394,  2395,  2396,  2397,  2398,  2399,  2401,  2406,
    2410,  2415,  2421,  2430,  2431,  2433,  2438,  2445,  2446,  2447,
    2448,  2449,  2450,  2451,  2452,  2453,  2454,  2455,  2456,  2461,
    2464,  2467,  2470,  2473,  2479,  2485,  2490,  2490,  2500,  2499,
    2543,  2542,  2594,  2595,  2599,  2606,  2607,  2611,  2619,  2618,
    2668,  2673,  2680,  2687,  2697,  2698,  2702,  2710,  2711,  2715,
    2724,  2725,  2726,  2734,  2735,  2739,  2740,  2743,  2744,  2747,
    2753,  2760,  2761
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
  "SWITCH", "CASE", "DEFAULT", "SELECT", "SELF", "FSELF", "TRY", "CATCH",
  "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL", "INIT", "LOAD",
  "LAUNCH", "CONST_KW", "EXPORT", "IMPORT", "DIRECTIVE", "COLON",
  "FUNCDECL", "STATIC", "INNERFUNC", "FORDOT", "LISTPAR", "LOOP", "ENUM",
  "TRUE_TOKEN", "FALSE_TOKEN", "STATE", "OUTER_STRING", "CLOSEPAR",
  "OPENPAR", "CLOSESQUARE", "OPENSQUARE", "DOT", "OPEN_GRAPH",
  "CLOSE_GRAPH", "ARROW", "VBAR", "ASSIGN_POW", "ASSIGN_SHL", "ASSIGN_SHR",
  "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD", "ASSIGN_DIV",
  "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_AS", "OP_TO",
  "COMMA", "QUESTION", "OR", "AND", "NOT", "LE", "GE", "LT", "GT", "NEQ",
  "EEQ", "PROVIDES", "OP_NOTIN", "OP_IN", "DIESIS", "ATSIGN", "CAP_CAP",
  "VBAR_VBAR", "AMPER_AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR",
  "POW", "SHR", "SHL", "CAP_XOROOB", "CAP_ISOOB", "CAP_DEOOB", "CAP_OOB",
  "CAP_EVAL", "TILDE", "NEG", "AMPER", "DECREMENT", "INCREMENT", "DOLLAR",
  "$accept", "input", "body", "line", "toplevel_statement",
  "INTNUM_WITH_MINUS", "load_statement", "statement", "base_statement",
  "assignment_def_list", "def_statement", "while_statement", "@1",
  "while_decl", "while_short_decl", "loop_statement", "@2",
  "loop_terminator", "if_statement", "@3", "if_decl", "if_short_decl",
  "elif_or_else", "@4", "else_decl", "elif_statement", "@5", "elif_decl",
  "statement_list", "break_statement", "continue_statement",
  "forin_statement", "@6", "@7", "forin_rest", "for_to_expr",
  "for_to_step_clause", "forin_statement_list", "forin_statement_elem",
  "fordot_statement", "self_print_statement", "outer_print_statement",
  "first_loop_block", "@8", "last_loop_block", "@9", "middle_loop_block",
  "@10", "switch_statement", "@11", "switch_decl", "case_list",
  "case_statement", "@12", "@13", "@14", "@15", "default_statement", "@16",
  "default_decl", "default_body", "case_expression_list", "case_element",
  "select_statement", "@17", "select_decl", "selcase_list",
  "selcase_statement", "@18", "@19", "@20", "@21",
  "selcase_expression_list", "selcase_element", "try_statement", "@22",
  "try_decl", "catch_statements", "catch_list", "catch_statement",
  "catch_decl", "catchcase_element_list", "catchcase_element",
  "raise_statement", "func_statement", "func_decl", "@23",
  "func_decl_short", "@24", "func_begin", "param_list", "param_symbol",
  "static_block", "@25", "@26", "static_decl", "static_short_decl",
  "launch_statement", "const_statement", "export_statement",
  "export_symbol_list", "import_statement", "attribute_statement",
  "import_symbol_list", "directive_statement", "directive_pair_list",
  "directive_pair", "class_decl", "@27", "class_def_inner",
  "class_param_list", "@28", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "class_statement_list",
  "class_statement", "init_decl", "@29", "property_decl", "state_decl",
  "state_statements", "enum_statement", "@30", "enum_statement_list",
  "enum_item_decl", "enum_item_terminator", "object_decl", "@31",
  "object_decl_inner", "object_statement_list", "object_statement",
  "global_statement", "@32", "global_symbol_list", "globalized_symbol",
  "return_statement", "const_atom_non_minus", "const_atom",
  "atomic_symbol", "var_atom", "OPT_EOL", "expression", "range_decl",
  "func_call", "@33", "nameless_func", "@34", "nameless_block", "@35",
  "nameless_func_decl_inner", "nameless_block_decl_inner", "innerfunc",
  "@36", "iif_expr", "array_decl", "dotarray_decl", "dict_decl",
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
     365,   366,   367,   368,   369,   370
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   116,   117,   118,   118,   119,   119,   119,   120,   120,
     120,   120,   120,   120,   120,   120,   120,   120,   121,   121,
     122,   122,   122,   123,   123,   123,   123,   123,   123,   123,
     124,   124,   124,   124,   124,   124,   124,   124,   124,   124,
     124,   124,   124,   124,   124,   124,   124,   124,   124,   124,
     125,   125,   125,   126,   126,   128,   127,   127,   129,   129,
     130,   130,   132,   131,   131,   131,   133,   133,   135,   134,
     134,   136,   136,   137,   137,   138,   138,   139,   138,   140,
     140,   142,   141,   143,   143,   144,   144,   145,   145,   146,
     146,   146,   148,   147,   149,   147,   147,   147,   150,   150,
     151,   151,   151,   152,   152,   152,   153,   153,   154,   154,
     154,   154,   155,   155,   156,   156,   156,   156,   156,   156,
     157,   159,   158,   158,   158,   161,   160,   160,   160,   163,
     162,   162,   162,   165,   164,   166,   166,   167,   167,   167,
     168,   169,   168,   170,   168,   171,   168,   172,   168,   173,
     174,   173,   175,   175,   175,   176,   176,   177,   177,   178,
     178,   178,   178,   178,   180,   179,   181,   181,   182,   182,
     182,   183,   184,   183,   185,   183,   186,   183,   187,   183,
     188,   188,   189,   189,   189,   190,   191,   190,   192,   192,
     193,   193,   194,   194,   195,   196,   196,   196,   196,   196,
     197,   197,   198,   198,   199,   199,   200,   200,   201,   202,
     201,   201,   203,   204,   203,   205,   206,   206,   206,   207,
     208,   209,   208,   210,   208,   211,   211,   212,   212,   213,
     213,   214,   214,   214,   215,   215,   215,   216,   216,   217,
     217,   217,   217,   217,   217,   217,   217,   217,   217,   217,
     217,   217,   217,   218,   218,   219,   219,   220,   220,   221,
     221,   222,   222,   222,   224,   223,   225,   225,   226,   226,
     227,   226,   228,   228,   229,   229,   230,   231,   231,   231,
     232,   232,   233,   233,   233,   233,   233,   233,   235,   234,
     236,   236,   237,   238,   238,   240,   239,   241,   241,   242,
     242,   242,   242,   243,   243,   245,   244,   246,   246,   247,
     247,   248,   248,   248,   248,   248,   250,   249,   251,   251,
     251,   251,   251,   252,   253,   253,   253,   254,   254,   254,
     254,   254,   254,   255,   255,   255,   255,   255,   255,   256,
     257,   257,   257,   258,   258,   259,   259,   259,   259,   259,
     259,   259,   259,   259,   259,   259,   259,   259,   259,   259,
     259,   259,   259,   259,   259,   259,   259,   259,   259,   259,
     259,   259,   259,   259,   259,   259,   259,   259,   259,   259,
     259,   259,   259,   259,   259,   259,   259,   259,   259,   259,
     259,   259,   259,   259,   259,   259,   259,   259,   259,   259,
     259,   259,   259,   259,   259,   259,   259,   259,   259,   259,
     259,   259,   259,   259,   259,   259,   259,   259,   259,   260,
     260,   260,   260,   260,   261,   261,   262,   261,   264,   263,
     266,   265,   267,   267,   267,   268,   268,   268,   270,   269,
     271,   271,   271,   271,   272,   272,   272,   273,   273,   273,
     274,   274,   274,   275,   275,   276,   276,   277,   277,   278,
     278,   279,   279
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     1,     3,     3,     3,     2,
       2,     4,     6,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     5,     3,     3,     0,     5,     2,     3,     3,
       3,     3,     0,     7,     3,     3,     0,     1,     0,     6,
       2,     3,     3,     3,     3,     0,     1,     0,     3,     2,
       3,     0,     4,     3,     3,     0,     2,     2,     3,     2,
       3,     3,     0,     6,     0,     6,     5,     3,     2,     4,
       4,     4,     3,     0,     2,     2,     0,     2,     1,     1,
       1,     1,     3,     3,     3,     2,     3,     2,     3,     3,
       1,     0,     6,     3,     3,     0,     6,     3,     3,     0,
       6,     3,     3,     0,     6,     3,     3,     0,     2,     3,
       1,     0,     5,     0,     5,     0,     5,     0,     5,     0,
       0,     3,     0,     1,     2,     2,     2,     1,     3,     1,
       1,     1,     3,     1,     0,     6,     3,     3,     0,     2,
       3,     1,     0,     5,     0,     5,     0,     5,     0,     5,
       1,     3,     0,     1,     1,     3,     0,     6,     2,     3,
       0,     1,     1,     2,     2,     2,     4,     3,     5,     3,
       1,     3,     1,     1,     3,     3,     5,     2,     5,     0,
       7,     3,     5,     0,     6,     2,     0,     1,     3,     1,
       0,     0,     5,     0,     3,     2,     3,     2,     3,     3,
       3,     5,     5,     3,     2,     3,     3,     1,     3,     3,
       5,     5,     7,     7,     7,     7,     4,     4,     4,     4,
       6,     6,     3,     4,     4,     1,     3,     3,     3,     1,
       3,     3,     3,     3,     0,     7,     3,     2,     0,     3,
       0,     5,     0,     2,     1,     3,     2,     0,     2,     3,
       0,     2,     1,     1,     1,     1,     1,     1,     0,     7,
       5,     4,     6,     1,     2,     0,     7,     0,     2,     1,
       4,     1,     2,     1,     1,     0,     7,     2,     2,     0,
       2,     1,     1,     1,     1,     1,     0,     4,     1,     1,
       2,     3,     3,     1,     2,     3,     3,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     0,     1,     1,     1,     2,     2,     2,
       3,     3,     3,     2,     3,     4,     4,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     2,     3,     2,     2,
       2,     2,     3,     3,     3,     3,     3,     3,     3,     2,
       3,     3,     3,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     1,     1,     1,     1,     1,     1,     2,     1,
       4,     5,     3,     1,     1,     3,     5,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       4,     4,     5,     7,     4,     3,     0,     6,     0,     6,
       0,     6,     4,     3,     2,     2,     2,     2,     0,     6,
       5,     5,     4,     3,     2,     3,     3,     2,     3,     3,
       3,     3,     4,     1,     3,     1,     3,     0,     1,     1,
       3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    25,   330,   331,   339,   332,
     327,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   341,   342,     0,     0,     0,     0,     0,   316,     0,
       0,     0,     0,     0,     0,     0,   438,     0,     0,     0,
       0,   328,   329,   120,     0,     0,   430,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     4,     5,     8,    14,    23,    33,    34,
      55,     0,    35,    39,    68,     0,    40,    41,    36,    47,
      48,    49,    37,   133,    38,   164,    42,   186,    43,    10,
     220,     0,     0,    46,    15,    16,    17,    24,     9,    11,
      13,    12,    45,    44,   345,   340,   346,   453,   404,   395,
     392,   393,   394,   396,   399,   397,   403,     0,    29,     0,
       0,     6,     0,   339,     0,    50,     0,   339,   428,     0,
       0,    87,     0,    89,     0,     0,     0,     0,   459,     0,
       0,     0,     0,     0,     0,     0,   188,     0,     0,     0,
       0,   264,     0,   305,     0,   324,     0,     0,     0,     0,
       0,     0,     0,   395,     0,     0,     0,   234,   237,     0,
       0,     0,     0,     0,     0,     0,     0,   259,     0,   215,
       0,     0,     0,     0,   447,   455,     0,     0,    62,     0,
     295,     0,     0,   444,     0,   453,     0,     0,     0,   379,
       0,   117,   453,     0,   386,   385,   353,     0,   115,     0,
     391,   390,   389,   388,   387,   366,   348,   347,   349,     0,
     371,   369,   384,   383,    85,     0,     0,     0,    57,    85,
      70,   137,   168,    85,     0,    85,   221,   223,   207,     0,
       0,    30,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   343,
     343,   343,   343,   343,   343,   343,   343,   343,   343,   343,
     370,   368,   398,     0,     0,     0,    18,   337,   338,   333,
     334,   335,     0,   336,     0,   354,    54,    53,     0,     0,
      59,    61,    58,    60,    88,    91,    90,    72,    74,    71,
      73,    97,     0,     0,     0,   136,   135,     7,   167,   166,
     189,   185,   205,   204,    28,     0,    27,     0,   326,   325,
     319,   323,     0,     0,    22,    20,    21,   230,   229,   233,
       0,   236,   235,     0,   252,     0,     0,     0,     0,   239,
       0,     0,   258,     0,   257,     0,    26,     0,   216,   220,
     220,   113,   112,   449,   448,   458,     0,    65,    85,    64,
       0,   418,   419,     0,   450,     0,     0,   446,   445,     0,
     451,     0,     0,   219,     0,   217,   220,   119,   116,   118,
     114,   351,   350,   352,     0,     0,     0,     0,     0,     0,
     225,   227,     0,    85,     0,   211,   213,     0,   425,     0,
       0,     0,   402,   412,   416,   417,   415,   414,   413,   411,
     410,   409,   408,   407,   405,   443,     0,   378,   377,   376,
     375,   374,   373,   367,   372,   382,   381,   380,   344,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   454,   254,    19,   253,     0,    51,    94,     0,   460,
       0,    92,     0,   216,   280,   272,     0,     0,     0,   309,
     317,     0,   320,     0,     0,   238,   246,   248,     0,   249,
       0,   247,     0,     0,   256,   261,   262,   263,   260,   434,
       0,    85,    85,   456,     0,   297,   421,   420,     0,   461,
     452,     0,   437,   436,   435,     0,    85,     0,    86,     0,
       0,     0,    77,    76,    81,     0,   140,     0,     0,   138,
       0,   150,     0,   171,     0,     0,   169,     0,     0,   191,
     192,    85,   226,   228,     0,     0,   224,     0,   209,     0,
     426,   424,     0,   400,     0,   442,     0,   363,   362,   361,
     356,   355,   359,   358,   357,   360,   365,   364,    31,     0,
       0,     0,     0,    96,     0,   267,     0,     0,     0,   308,
     277,   273,   274,   307,     0,   322,   321,   232,   231,     0,
       0,   240,     0,     0,   241,     0,     0,   433,     0,     0,
       0,    66,     0,     0,   422,     0,   218,     0,    56,     0,
      79,     0,     0,     0,    85,    85,   139,     0,   163,   161,
     159,   160,     0,   157,   154,     0,     0,   170,     0,   183,
     184,     0,   180,     0,     0,   195,   202,   203,     0,     0,
     200,     0,   193,     0,   206,     0,     0,     0,   208,   212,
       0,   401,   406,   441,   440,     0,    52,     0,     0,    95,
     102,     0,    93,   270,   269,   282,     0,     0,     0,     0,
       0,     0,   283,   286,   281,   285,   284,   287,   266,     0,
     276,     0,   311,     0,   312,   315,   314,   313,   310,   250,
     251,     0,     0,     0,     0,   432,   429,   439,     0,    67,
     299,     0,     0,   301,   298,     0,   462,   431,    80,    84,
      83,    69,     0,     0,   145,   147,     0,   141,   143,     0,
     134,    85,     0,   151,   176,   178,   172,   174,   182,   165,
     199,     0,   197,     0,     0,   187,   222,   214,     0,   427,
      32,     0,     0,     0,   108,     0,     0,   109,   110,   111,
      98,   101,     0,   100,     0,     0,   265,   288,     0,     0,
     278,     0,   275,   306,   242,   244,   243,   245,    63,   303,
       0,   304,   302,   296,   423,    82,    85,     0,   162,    85,
       0,   158,     0,   156,    85,     0,    85,     0,   181,   196,
     201,     0,   210,     0,   121,     0,     0,   125,     0,     0,
     129,     0,     0,   107,   105,   104,   271,     0,   220,     0,
       0,   279,     0,     0,   148,     0,   144,     0,   179,     0,
     175,   198,   124,    85,   123,   128,    85,   127,   132,    85,
     131,    99,   291,    85,     0,   293,     0,     0,   300,     0,
       0,     0,     0,   290,   216,     0,   294,     0,     0,     0,
       0,     0,   292,   122,   126,   130,   289,     0
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    63,    64,   293,    65,   508,    67,   124,
      68,    69,   224,    70,    71,    72,   368,   688,    73,   229,
      74,    75,   511,   604,   512,   513,   605,   514,   394,    76,
      77,    78,   564,   561,   649,   457,   743,   735,   736,    79,
      80,    81,   737,   813,   738,   816,   739,   819,    82,   231,
      83,   396,   519,   769,   770,   766,   767,   520,   616,   521,
     713,   612,   613,    84,   232,    85,   397,   526,   776,   777,
     774,   775,   621,   622,    86,   233,    87,   528,   529,   530,
     531,   629,   630,    88,    89,    90,   637,    91,   537,    92,
     384,   385,   235,   403,   404,   236,   237,    93,    94,    95,
     169,    96,    97,   173,    98,   176,   177,    99,   325,   464,
     465,   744,   468,   571,   572,   670,   567,   664,   665,   798,
     666,   667,   827,   100,   370,   592,   694,   762,   101,   327,
     469,   574,   678,   102,   157,   332,   333,   103,   104,   294,
     105,   106,   439,   107,   108,   109,   640,   110,   180,   111,
     198,   359,   386,   112,   181,   113,   114,   115,   116,   117,
     186,   366,   139,   197
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -745
static const yytype_int16 yypact[] =
{
    -745,    14,   787,  -745,    72,  -745,  -745,  -745,   238,  -745,
    -745,   117,   308,  3605,   474,   466,  3677,   442,  3749,    56,
    3821,  -745,  -745,    28,  3893,   452,   548,  3389,  -745,   359,
    3965,   554,   470,   290,   562,   276,  -745,  4037,  5442,   369,
     146,  -745,  -745,  -745,  5802,  5297,  -745,  5802,  3461,  5802,
    5802,  5802,  3533,  5802,  5802,  5802,  5802,  5802,  5802,   329,
    5802,  5802,   529,  -745,  -745,  -745,  -745,  -745,  -745,  -745,
    -745,  3317,  -745,  -745,  -745,  3317,  -745,  -745,  -745,  -745,
    -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,
     119,  3317,   210,  -745,  -745,  -745,  -745,  -745,  -745,  -745,
    -745,  -745,  -745,  -745,  -745,  -745,  -745,  4839,  -745,  -745,
    -745,  -745,  -745,  -745,  -745,  -745,  -745,   230,  -745,    54,
    5802,  -745,   286,  -745,   131,   217,   277,   236,  -745,  4664,
     368,  -745,   413,  -745,   451,   316,  4737,   465,   259,   331,
     471,  4890,   522,   534,  4941,   549,  -745,  3317,   553,  4992,
     556,  -745,   559,  -745,   566,  -745,  5043,   565,   589,   595,
     608,   613,  6345,   621,   625,   497,   627,  -745,  -745,   134,
     628,    19,    17,   127,   629,   530,   135,  -745,   630,  -745,
     214,   214,   645,  5094,  -745,  6345,   569,   646,  -745,  3317,
    -745,  6037,  5514,  -745,   593,  5862,     8,    75,   130,  6590,
     650,  -745,  6345,   144,   393,   393,   203,   651,  -745,   154,
     203,   203,   203,   203,   203,   203,  -745,  -745,  -745,   289,
     203,   203,  -745,  -745,  -745,   618,   654,   228,  -745,  -745,
    -745,  -745,  -745,  -745,   372,  -745,  -745,  -745,  -745,   653,
     138,  -745,  5586,  5370,   657,  5802,  5802,  5802,  5802,  5802,
    5802,  5802,  5802,  5802,  5802,  5802,  5802,  4109,  5802,  5802,
    5802,  5802,  5802,  5802,  5802,  5802,   659,  5802,  5802,   664,
     664,   664,   664,   664,   664,   664,   664,   664,   664,   664,
    -745,  -745,  -745,  5802,  5802,   665,  -745,  -745,  -745,  -745,
    -745,  -745,   656,  -745,   666,  6345,  -745,  -745,   667,  5802,
    -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,
    -745,  -745,  5802,   667,  4181,  -745,  -745,  -745,  -745,  -745,
    -745,  -745,  -745,  -745,  -745,   262,  -745,   469,  -745,  -745,
    -745,  -745,   176,    93,  -745,  -745,  -745,  -745,  -745,  -745,
      59,  -745,  -745,   674,  -745,   668,   197,   199,   669,  -745,
     616,   679,  -745,    84,  -745,   680,  -745,   684,   682,   119,
     119,  -745,  -745,  -745,  -745,  -745,  5802,  -745,  -745,  -745,
     686,  -745,  -745,  6088,  -745,  5658,  5802,  -745,  -745,   633,
    -745,  5802,   631,  -745,    71,  -745,   119,  -745,  -745,  -745,
    -745,  -745,  -745,  -745,  1822,  1477,   417,   499,  1592,   407,
    -745,  -745,  1937,  -745,  3317,  -745,  -745,   139,  -745,   140,
    5802,  5924,  -745,  6345,  6345,  6345,  6345,  6345,  6345,  6345,
    6345,  6345,  6345,  6345,  6394,  -745,  4591,  6541,  6590,   248,
     248,   248,   248,   248,   248,  -745,   393,   393,  -745,  5802,
    5802,  5802,  5802,  5802,  5802,  5802,  5802,  5802,  5802,  5802,
    4788,  6443,  -745,  -745,  -745,   617,  6345,  -745,  6139,  -745,
     688,  6345,   691,   682,  -745,   638,   692,   690,   694,  -745,
    -745,   598,  -745,   695,   696,  -745,  -745,  -745,   697,  -745,
     698,  -745,     2,    27,  -745,  -745,  -745,  -745,  -745,  -745,
     141,  -745,  -745,  6345,  2052,  -745,  -745,  -745,  5986,  6345,
    -745,  6192,  -745,  -745,  -745,   682,  -745,   699,  -745,   579,
    4253,   700,  -745,  -745,  -745,   702,  -745,    73,   422,  -745,
     703,  -745,   704,  -745,   123,   706,  -745,    81,   707,   671,
    -745,  -745,  -745,  -745,   705,  2167,  -745,   658,  -745,   414,
    -745,  -745,  6243,  -745,  5802,  -745,  4325,   227,   227,   487,
     427,   427,   165,   165,   165,   254,   203,   203,  -745,  5802,
    5802,   439,  4397,  -745,   439,  -745,   142,   402,   708,  -745,
     661,   623,  -745,  -745,   558,  -745,  -745,  -745,  -745,   715,
     717,  -745,   718,   719,  -745,   720,   722,  -745,   728,  2282,
    2397,  5802,   398,  5802,  -745,  5802,  -745,  2512,  -745,   729,
    -745,   732,  5145,   733,  -745,  -745,  -745,   460,  -745,  -745,
    -745,   644,   112,  -745,  -745,   734,   496,  -745,   501,  -745,
    -745,   120,  -745,   735,   736,  -745,  -745,  -745,   667,    44,
    -745,   737,  -745,  1707,  -745,   738,   663,   687,  -745,  -745,
     689,  -745,   670,  -745,  6492,   186,  6345,   902,  3317,  -745,
    -745,  4529,  -745,  -745,  -745,  -745,   -22,   740,   742,   741,
     743,   744,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  5730,
    -745,   690,  -745,   748,  -745,  -745,  -745,  -745,  -745,  -745,
    -745,   749,   750,   751,   752,  -745,  -745,  -745,   753,  6345,
    -745,   182,   754,  -745,  -745,  6294,  6345,  -745,  -745,  -745,
    -745,  -745,  2627,  1477,  -745,  -745,    12,  -745,  -745,    94,
    -745,  -745,  3317,  -745,  -745,  -745,  -745,  -745,   602,  -745,
    -745,   756,  -745,   606,   667,  -745,  -745,  -745,   758,  -745,
    -745,   478,   535,   550,  -745,   714,   902,  -745,  -745,  -745,
    -745,  -745,  4469,  -745,   709,  5802,  -745,  -745,   710,   759,
    -745,   -44,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,
     109,  -745,  -745,  -745,  -745,  -745,  -745,  3317,  -745,  -745,
    3317,  -745,  2742,  -745,  -745,  3317,  -745,  3317,  -745,  -745,
    -745,   760,  -745,   764,  -745,  3317,   767,  -745,  3317,   768,
    -745,  3317,   771,  -745,  -745,  6345,  -745,  5196,   119,   109,
     757,  -745,   196,  1017,  -745,  1132,  -745,  1247,  -745,  1362,
    -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,
    -745,  -745,  -745,  -745,   774,  -745,   297,   267,  -745,  2857,
    2972,  3087,  3202,  -745,   682,   775,  -745,   776,   777,   778,
     779,   149,  -745,  -745,  -745,  -745,  -745,   781
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -745,  -745,  -745,  -745,  -745,  -303,  -745,    -2,  -745,  -745,
    -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,
    -745,  -745,    43,  -745,  -745,  -745,  -745,  -745,   -47,  -745,
    -745,  -745,  -745,  -745,   239,  -745,  -745,    68,  -745,  -745,
    -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,  -745,
    -745,  -745,  -745,  -745,  -745,  -745,  -745,   409,  -745,  -745,
    -745,  -745,    98,  -745,  -745,  -745,  -745,  -745,  -745,  -745,
    -745,  -745,  -745,    90,  -745,  -745,  -745,  -745,  -745,   282,
    -745,  -745,    96,  -745,  -277,  -744,  -745,  -745,  -745,  -732,
    -237,   311,  -332,  -745,  -745,  -745,  -745,  -745,  -745,  -745,
    -745,  -745,  -364,  -745,  -745,  -745,   468,  -745,  -745,  -745,
    -745,  -745,   367,  -745,   159,  -745,  -745,  -745,   266,  -745,
     268,  -745,  -745,  -745,  -745,  -745,  -745,    42,  -745,  -745,
    -745,  -745,  -745,  -745,  -745,  -745,   375,  -745,  -745,  -321,
     -10,  -745,   365,   -12,   -37,   818,  -745,  -745,  -745,  -745,
    -745,   672,  -745,  -745,  -745,  -745,  -745,  -745,  -745,   -35,
    -745,  -745,  -745,  -745
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -458
static const yytype_int16 yytable[] =
{
      66,   129,   125,   407,   136,   581,   141,   138,   144,   377,
     196,   801,   149,   203,     3,   156,   286,   209,   162,   474,
     345,   119,  -255,   346,   347,   183,   185,   491,   492,   145,
     584,   146,   191,   195,   284,   199,   202,   204,   205,   206,
     202,   210,   211,   212,   213,   214,   215,   722,   220,   221,
     487,  -255,   223,   745,   506,   285,   825,   142,   286,   287,
     473,   288,   289,   286,   287,   378,   288,   289,   826,   228,
     282,   147,   503,   230,   607,   118,   379,   286,   582,   608,
     609,   610,   624,   836,   625,   626,   284,   627,   286,   238,
     485,   486,   282,   583,   472,   826,  -318,  -255,   286,   282,
     608,   609,   610,   585,   282,   290,   291,   282,   295,   292,
     290,   291,   282,   286,   287,   707,   288,   289,   586,   282,
     121,   490,   723,   716,   618,   282,  -182,   619,   348,   620,
     349,   382,   380,   504,   297,   724,   383,   342,   354,   406,
     538,   540,   587,   653,   383,   321,   282,   388,   282,   505,
     538,   292,   190,   381,   282,   708,   292,   390,   282,   350,
     290,   291,   282,   717,   234,   282,  -182,   282,   282,   282,
     292,  -318,   628,   282,   282,   282,   282,   282,   282,   470,
     373,   292,   395,   282,   282,   759,   398,   369,   402,   730,
     709,   292,  -216,  -216,   539,   541,   588,   654,   718,   759,
     477,  -182,   479,   663,   847,   351,   292,   409,  -216,   298,
     675,   239,   343,   355,   611,   357,  -216,   505,   284,   505,
     505,   242,   284,   243,   244,   119,   566,   505,   693,   178,
     202,   411,   284,   413,   414,   415,   416,   417,   418,   419,
     420,   421,   422,   423,   424,   426,   427,   428,   429,   430,
     431,   432,   433,   434,   471,   436,   437,   760,   282,   242,
     761,   243,   244,   462,   284,  -268,   240,   277,   278,   279,
     358,   450,   451,   478,   761,   480,   835,   178,   280,   281,
     300,   119,   179,   242,  -428,   243,   244,   456,   455,   296,
     662,   170,   299,   391,  -268,   392,   171,   674,   239,   120,
     458,   120,   461,   459,   242,   283,   243,   244,   284,   122,
     242,   659,   243,   244,   123,   393,   280,   281,   463,   307,
     301,   494,   172,   271,   272,   273,   274,   275,   276,   277,
     278,   279,  -428,   216,   312,   217,   282,   266,   267,   268,
     280,   281,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   834,   493,   218,   535,   278,   279,   308,
     158,   280,   281,   498,   499,   159,   160,   280,   281,   501,
     187,   304,   188,   399,   282,   400,   282,   282,   282,   282,
     282,   282,   282,   282,   282,   282,   282,   282,   219,   282,
     282,   282,   282,   282,   282,   282,   282,   282,   542,   282,
     282,   690,   536,   768,   691,   655,   611,   692,   656,   313,
     532,   657,   189,   282,   282,   401,   305,   638,   515,   282,
     516,   282,   314,   614,   282,  -153,  -149,   547,   548,   549,
     550,   551,   552,   553,   554,   555,   556,   557,   658,   802,
     517,   518,   647,   137,   589,   590,   659,   660,   123,   242,
     533,   243,   244,   150,   306,   661,   282,   639,   151,   597,
    -152,   282,   282,   704,   282,  -153,   823,   132,   311,   133,
     466,   166,  -272,   167,   315,   130,   168,   131,   824,   783,
     134,   784,   648,   242,   633,   243,   244,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   602,   711,
     522,   467,   523,   705,   714,   282,   280,   281,  -149,   642,
     282,   282,   282,   282,   282,   282,   282,   282,   282,   282,
     282,   785,   524,   518,   645,   317,   274,   275,   276,   277,
     278,   279,   202,   222,   644,   123,   786,   318,   787,   712,
     280,   281,  -152,   242,   715,   243,   244,   202,   646,   152,
     651,   789,   320,   790,   153,   164,   322,   702,   703,   324,
     165,   672,   326,   174,   656,   282,   330,   673,   175,   328,
     363,   331,   340,  -457,  -457,  -457,  -457,  -457,   788,   689,
     599,   695,   600,   696,   272,   273,   274,   275,   276,   277,
     278,   279,   334,   791,   658,  -457,  -457,   841,   335,   575,
     280,   281,   659,   660,   331,   353,   619,   282,   620,   282,
     626,   336,   627,  -457,   282,  -457,   337,  -457,   721,   150,
    -457,  -457,   482,   483,   338,  -457,   364,  -457,   339,  -457,
     341,   344,   352,   356,   751,   440,   441,   442,   443,   444,
     445,   446,   447,   448,   449,   734,   740,   365,   361,   367,
     374,  -457,   282,   387,   389,   152,   405,   202,   282,   282,
     453,  -457,  -457,   412,   772,   435,  -457,   438,   452,   454,
     467,   476,   481,   123,  -457,  -457,  -457,  -457,  -457,  -457,
     475,  -457,  -457,  -457,  -457,   484,   175,   489,   383,   495,
     500,   563,   560,   502,   565,   569,   570,   573,   577,   578,
     527,   671,   598,   579,   580,   606,   727,   617,   634,   603,
     773,   668,   615,   636,   781,   623,   631,   669,   679,   803,
     680,   706,   805,   792,   681,   682,   683,   807,   684,   809,
     795,   685,   698,   797,   734,   699,   701,   710,   719,   720,
     725,   726,   728,   746,   729,   747,   765,   179,   284,   748,
     749,   753,   754,   755,   756,   757,   758,   763,   282,   779,
     282,   782,   800,   811,   796,   804,   829,   812,   806,   830,
     815,   818,   831,   808,   821,   810,   832,   833,   842,   843,
     844,   845,   846,   814,   638,   799,   817,    -2,     4,   820,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,   659,    16,   652,   793,    17,   525,   771,   778,    18,
      19,   632,    20,    21,    22,    23,   596,    24,    25,   780,
      26,    27,    28,   488,    29,    30,    31,    32,    33,    34,
     752,    35,   568,    36,    37,    38,    39,    40,    41,    42,
     676,    43,   677,    44,   828,    45,   576,    46,   163,     0,
       0,     0,     0,   360,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,  -106,    12,    13,    14,    15,     0,    16,     0,     0,
      17,   731,   732,   733,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,  -146,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
    -146,  -146,    20,    21,    22,    23,     0,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
    -146,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,  -142,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,  -142,  -142,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,  -142,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,  -177,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
    -177,  -177,    20,    21,    22,    23,     0,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
    -177,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,  -173,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,  -173,  -173,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,  -173,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,   -75,    12,    13,    14,
      15,     0,    16,   509,   510,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,     0,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
       0,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,  -190,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,   527,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,  -194,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,  -194,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
       0,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,   507,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,   534,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,     0,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
       0,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,   591,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,   635,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,     0,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
       0,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,   686,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,   687,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,     0,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
       0,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,     0,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,   697,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,   -78,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,     0,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
       0,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,  -155,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,   837,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,     0,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
       0,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,   838,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,   839,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,     0,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
       0,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,   840,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,     0,    43,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,    48,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,    52,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,     4,     0,
       5,     6,     7,     8,     9,    10,     0,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,     0,    24,   225,     0,
     226,    27,    28,     0,     0,    30,     0,     0,     0,     0,
       0,   227,     0,    36,    37,    38,    39,     0,    41,    42,
       0,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     154,     0,   155,     6,     7,   127,     9,    10,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,   128,     0,    36,     0,    38,     0,     0,
      41,    42,     0,     0,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   200,     0,   201,     6,     7,   127,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,   128,     0,    36,     0,    38,
       0,     0,    41,    42,     0,     0,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   207,     0,   208,     6,     7,   127,
       9,    10,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,    21,
      22,     0,     0,     0,     0,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,   128,     0,    36,
       0,    38,     0,     0,    41,    42,     0,     0,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   126,     0,     0,     6,
       7,   127,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,     0,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,   128,
       0,    36,     0,    38,     0,     0,    41,    42,     0,     0,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   135,     0,
       0,     6,     7,   127,     9,    10,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,    21,    22,     0,     0,     0,     0,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,   128,     0,    36,     0,    38,     0,     0,    41,    42,
       0,     0,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     140,     0,     0,     6,     7,   127,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,   128,     0,    36,     0,    38,     0,     0,
      41,    42,     0,     0,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   143,     0,     0,     6,     7,   127,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,   128,     0,    36,     0,    38,
       0,     0,    41,    42,     0,     0,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   148,     0,     0,     6,     7,   127,
       9,    10,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,    21,
      22,     0,     0,     0,     0,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,   128,     0,    36,
       0,    38,     0,     0,    41,    42,     0,     0,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   161,     0,     0,     6,
       7,   127,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,     0,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,   128,
       0,    36,     0,    38,     0,     0,    41,    42,     0,     0,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   182,     0,
       0,     6,     7,   127,     9,    10,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,    21,    22,     0,     0,     0,     0,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,   128,     0,    36,     0,    38,     0,     0,    41,    42,
       0,     0,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     425,     0,     0,     6,     7,   127,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,   128,     0,    36,     0,    38,     0,     0,
      41,    42,     0,     0,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   460,     0,     0,     6,     7,   127,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,   128,     0,    36,     0,    38,
       0,     0,    41,    42,     0,     0,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   601,     0,     0,     6,     7,   127,
       9,    10,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,    21,
      22,     0,     0,     0,     0,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,   128,     0,    36,
       0,    38,     0,     0,    41,    42,     0,     0,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   643,     0,     0,     6,
       7,   127,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,     0,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,   128,
       0,    36,     0,    38,     0,     0,    41,    42,     0,     0,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   650,     0,
       0,     6,     7,   127,     9,    10,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,    21,    22,     0,     0,     0,     0,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,   128,     0,    36,     0,    38,     0,     0,    41,    42,
       0,     0,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     794,     0,     0,     6,     7,   127,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,   128,     0,    36,     0,    38,     0,     0,
      41,    42,     0,     0,     0,    44,     0,    45,     0,    46,
     741,     0,  -103,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,  -103,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,   242,     0,   243,   244,     0,
       0,     0,   545,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,     0,     0,   742,   257,   258,
     259,     0,   260,   261,   262,   263,   264,   265,   266,   267,
     268,     0,     0,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   546,     0,     0,     0,     0,     0,
       0,     0,   280,   281,     0,     0,     0,   242,     0,   243,
     244,     0,     0,     0,     0,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   302,     0,     0,
     257,   258,   259,     0,   260,   261,   262,   263,   264,   265,
     266,   267,   268,     0,     0,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,     0,     0,     0,     0,
       0,     0,     0,     0,   280,   281,     0,   303,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     242,     0,   243,   244,     0,     0,     0,     0,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
     309,     0,     0,   257,   258,   259,     0,   260,   261,   262,
     263,   264,   265,   266,   267,   268,     0,     0,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,     0,
       0,     0,     0,     0,     0,     0,     0,   280,   281,     0,
     310,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   558,     0,   242,     0,   243,   244,     0,     0,     0,
       0,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,     0,     0,     0,   257,   258,   259,     0,
     260,   261,   262,   263,   264,   265,   266,   267,   268,     0,
       0,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,   241,     0,   242,     0,   243,   244,     0,     0,
     280,   281,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,     0,     0,   559,   257,   258,   259,
       0,   260,   261,   262,   263,   264,   265,   266,   267,   268,
       0,     0,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   316,     0,   242,     0,   243,   244,     0,
       0,   280,   281,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,     0,     0,     0,   257,   258,
     259,     0,   260,   261,   262,   263,   264,   265,   266,   267,
     268,     0,     0,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   319,     0,   242,     0,   243,   244,
       0,     0,   280,   281,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,     0,     0,     0,   257,
     258,   259,     0,   260,   261,   262,   263,   264,   265,   266,
     267,   268,     0,     0,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   323,     0,   242,     0,   243,
     244,     0,     0,   280,   281,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,     0,     0,     0,
     257,   258,   259,     0,   260,   261,   262,   263,   264,   265,
     266,   267,   268,     0,     0,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   329,     0,   242,     0,
     243,   244,     0,     0,   280,   281,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,     0,     0,
       0,   257,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   362,     0,   242,
       0,   243,   244,     0,     0,   280,   281,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,     0,
       0,     0,   257,   258,   259,     0,   260,   261,   262,   263,
     264,   265,   266,   267,   268,     0,     0,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   700,     0,
     242,     0,   243,   244,     0,     0,   280,   281,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
       0,     0,     0,   257,   258,   259,     0,   260,   261,   262,
     263,   264,   265,   266,   267,   268,     0,     0,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   822,
       0,   242,     0,   243,   244,     0,     0,   280,   281,   245,
     246,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,     0,     0,     0,   257,   258,   259,     0,   260,   261,
     262,   263,   264,   265,   266,   267,   268,     0,     0,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
       0,     0,   242,     0,   243,   244,     0,     0,   280,   281,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,     0,     0,     0,   257,   258,   259,     0,   260,
     261,   262,   263,   264,   265,   266,   267,   268,     0,     0,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,     6,     7,   127,     9,    10,     0,     0,     0,   280,
     281,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     192,   128,     0,    36,     0,    38,     0,     0,    41,    42,
       0,     0,     0,    44,   193,    45,     0,    46,     0,   194,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     6,     7,   127,     9,    10,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,    21,    22,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,   192,   128,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     6,     7,   127,     9,
      10,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,    21,    22,
       0,   410,     0,     0,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,   128,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,     0,    44,   184,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     6,     7,
     127,     9,    10,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
      21,    22,     0,     0,     0,     0,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,   128,     0,
      36,     0,    38,     0,     0,    41,    42,     0,     0,     0,
      44,   372,    45,     0,    46,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       6,     7,   127,     9,    10,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,    50,     0,     0,
       0,    51,    21,    22,     0,     0,     0,     0,     0,    53,
      54,    55,    56,    57,    58,     0,    59,    60,    61,    62,
     128,     0,    36,     0,    38,     0,     0,    41,    42,     0,
       0,   408,    44,     0,    45,     0,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     6,     7,   127,     9,    10,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,    50,
       0,     0,     0,    51,    21,    22,     0,     0,     0,     0,
       0,    53,    54,    55,    56,    57,    58,     0,    59,    60,
      61,    62,   128,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,     0,    44,   497,    45,     0,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     6,     7,   127,     9,    10,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,    50,     0,     0,     0,    51,    21,    22,     0,     0,
       0,     0,     0,    53,    54,    55,    56,    57,    58,     0,
      59,    60,    61,    62,   128,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,   750,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     6,     7,   127,     9,
      10,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,    21,    22,
       0,     0,     0,     0,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,   128,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,   375,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,   242,     0,
     243,   244,     0,     0,   376,     0,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,     0,     0,
       0,   257,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   375,     0,     0,
       0,     0,     0,     0,     0,   280,   281,     0,     0,     0,
     242,   543,   243,   244,     0,     0,     0,     0,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
       0,     0,     0,   257,   258,   259,     0,   260,   261,   262,
     263,   264,   265,   266,   267,   268,     0,     0,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   593,
       0,     0,     0,     0,     0,     0,     0,   280,   281,     0,
       0,     0,   242,   594,   243,   244,     0,     0,     0,     0,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,     0,     0,     0,   257,   258,   259,     0,   260,
     261,   262,   263,   264,   265,   266,   267,   268,     0,     0,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,     0,   371,   242,     0,   243,   244,     0,     0,   280,
     281,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,     0,     0,     0,   257,   258,   259,     0,
     260,   261,   262,   263,   264,   265,   266,   267,   268,     0,
       0,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,     0,     0,   242,   496,   243,   244,     0,     0,
     280,   281,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,     0,     0,     0,   257,   258,   259,
       0,   260,   261,   262,   263,   264,   265,   266,   267,   268,
       0,     0,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,     0,     0,   242,     0,   243,   244,     0,
       0,   280,   281,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,     0,   562,     0,   257,   258,
     259,     0,   260,   261,   262,   263,   264,   265,   266,   267,
     268,     0,     0,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,     0,     0,     0,     0,   242,     0,
     243,   244,   280,   281,   595,     0,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,     0,     0,
       0,   257,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,     0,     0,   242,
     641,   243,   244,     0,     0,   280,   281,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,     0,
       0,     0,   257,   258,   259,     0,   260,   261,   262,   263,
     264,   265,   266,   267,   268,     0,     0,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,     0,     0,
     242,   764,   243,   244,     0,     0,   280,   281,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
       0,     0,     0,   257,   258,   259,     0,   260,   261,   262,
     263,   264,   265,   266,   267,   268,     0,     0,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,     0,
       0,   242,     0,   243,   244,     0,     0,   280,   281,   245,
     246,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,     0,     0,     0,   257,   258,   259,     0,   260,   261,
     262,   263,   264,   265,   266,   267,   268,     0,     0,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     242,     0,   243,   244,     0,     0,     0,     0,   280,   281,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   256,
       0,     0,   544,   257,   258,   259,     0,   260,   261,   262,
     263,   264,   265,   266,   267,   268,     0,     0,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   242,
       0,   243,   244,     0,     0,     0,     0,   280,   281,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   257,   258,   259,     0,   260,   261,   262,   263,
     264,   265,   266,   267,   268,     0,     0,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   242,     0,
     243,   244,     0,     0,     0,     0,   280,   281,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   242,     0,   243,
     244,     0,     0,     0,     0,   280,   281,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   259,     0,   260,   261,   262,   263,   264,   265,
     266,   267,   268,     0,     0,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   242,     0,   243,   244,
       0,     0,     0,     0,   280,   281,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   260,   261,   262,   263,   264,   265,   266,
     267,   268,     0,     0,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,     0,     0,     0,     0,     0,
       0,     0,     0,   280,   281
};

static const yytype_int16 yycheck[] =
{
       2,    13,    12,   240,    16,     3,    18,    17,    20,     1,
      45,    55,    24,    48,     0,    27,     4,    52,    30,   340,
       1,    43,     3,     6,     7,    37,    38,   359,   360,     1,
       3,     3,    44,    45,    78,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,     3,    60,    61,
     353,    32,    62,    75,   386,     1,   800,     1,     4,     5,
       1,     7,     8,     4,     5,    57,     7,     8,   800,    71,
     107,    43,     1,    75,     1,     3,     1,     4,    76,     6,
       7,     8,     1,   827,     3,     4,    78,     6,     4,    91,
       6,     7,   129,    91,     1,   827,     3,    78,     4,   136,
       6,     7,     8,    76,   141,    51,    52,   144,   120,    97,
      51,    52,   149,     4,     5,     3,     7,     8,    91,   156,
       3,   358,    78,     3,     1,   162,     3,     4,     1,     6,
       3,     1,    57,    62,     3,    91,     6,     3,     3,     1,
       1,     1,     1,     1,     6,   147,   183,     3,   185,    78,
       1,    97,     6,    78,   191,    43,    97,     3,   195,    32,
      51,    52,   199,    43,    45,   202,    43,   204,   205,   206,
      97,    78,    91,   210,   211,   212,   213,   214,   215,     3,
     192,    97,   229,   220,   221,     3,   233,   189,   235,     3,
      78,    97,    62,    55,    55,    55,    55,    55,    78,     3,
       3,    78,     3,   567,    55,    78,    97,   242,    78,    78,
     574,     1,    78,    78,   517,     1,    78,    78,    78,    78,
      78,    56,    78,    58,    59,    43,   463,    78,   592,     1,
     242,   243,    78,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,    78,   267,   268,    75,   295,    56,
      78,    58,    59,     1,    78,     3,    56,   102,   103,   104,
      56,   283,   284,    76,    78,    76,     9,     1,   113,   114,
       3,    43,     6,    56,    56,    58,    59,   299,   298,     3,
     567,     1,    75,     4,    32,     6,     6,   574,     1,    63,
     312,    63,   314,   313,    56,    75,    58,    59,    78,     1,
      56,    44,    58,    59,     6,    26,   113,   114,    56,     3,
      43,   368,    32,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    56,     4,    75,     6,   373,    89,    90,    91,
     113,   114,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,    56,   366,    26,   403,   103,   104,    43,
       1,   113,   114,   375,   376,     6,     7,   113,   114,   381,
       1,     3,     3,     1,   411,     3,   413,   414,   415,   416,
     417,   418,   419,   420,   421,   422,   423,   424,    59,   426,
     427,   428,   429,   430,   431,   432,   433,   434,   410,   436,
     437,     3,   404,   706,     6,     3,   709,     9,     6,    78,
       3,     9,    43,   450,   451,    43,     3,     3,     1,   456,
       3,   458,    91,     1,   461,     3,     9,   439,   440,   441,
     442,   443,   444,   445,   446,   447,   448,   449,    36,   760,
      23,    24,     3,     1,   491,   492,    44,    45,     6,    56,
      43,    58,    59,     1,     3,    53,   493,    43,     6,   506,
      43,   498,   499,     3,   501,    43,   798,     1,     3,     3,
       1,     1,     3,     3,     3,     1,     6,     3,   799,     1,
      14,     3,    43,    56,   531,    58,    59,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   510,     3,
       1,    32,     3,    43,     3,   542,   113,   114,     9,   544,
     547,   548,   549,   550,   551,   552,   553,   554,   555,   556,
     557,    43,    23,    24,   559,     3,    99,   100,   101,   102,
     103,   104,   544,     4,   546,     6,     1,     3,     3,    43,
     113,   114,    43,    56,    43,    58,    59,   559,   560,     1,
     562,     1,     3,     3,     6,     1,     3,   604,   605,     3,
       6,     3,     3,     1,     6,   602,     1,     9,     6,     3,
       1,     6,    75,     4,     5,     6,     7,     8,    43,   591,
       1,   593,     3,   595,    97,    98,    99,   100,   101,   102,
     103,   104,     3,    43,    36,    26,    27,   834,     3,     1,
     113,   114,    44,    45,     6,    75,     4,   644,     6,   646,
       4,     3,     6,    44,   651,    46,     3,    48,   628,     1,
      51,    52,     6,     7,     3,    56,    57,    58,     3,    60,
       3,     3,     3,     3,   669,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   647,   648,    78,     3,     3,
      57,    82,   689,     3,     3,     1,     3,   669,   695,   696,
       4,    92,    93,     6,   711,     6,    97,     3,     3,     3,
      32,     3,     3,     6,   105,   106,   107,   108,   109,   110,
       6,   112,   113,   114,   115,     6,     6,     3,     6,     3,
      57,     3,    75,    62,     3,     3,     6,     3,     3,     3,
      29,    78,     3,     6,     6,     3,    43,     3,     3,     9,
     712,     3,     9,    55,   724,     9,     9,    56,     3,   766,
       3,    77,   769,     9,     6,     6,     6,   774,     6,   776,
     742,     3,     3,   745,   736,     3,     3,     3,     3,     3,
       3,     3,    55,     3,    55,     3,   703,     6,    78,     6,
       6,     3,     3,     3,     3,     3,     3,     3,   795,     3,
     797,     3,     3,     3,    55,   767,   813,     3,   770,   816,
       3,     3,   819,   775,     3,   777,   823,     3,     3,     3,
       3,     3,     3,   785,     3,    75,   788,     0,     1,   791,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    44,    15,   564,   736,    18,   397,   709,   718,    22,
      23,   529,    25,    26,    27,    28,   505,    30,    31,   723,
      33,    34,    35,   355,    37,    38,    39,    40,    41,    42,
     671,    44,   465,    46,    47,    48,    49,    50,    51,    52,
     574,    54,   574,    56,   802,    58,   471,    60,    30,    -1,
      -1,    -1,    -1,   181,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    19,    20,    21,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    -1,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      43,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    24,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    43,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    -1,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      43,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    24,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    43,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    16,    17,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    -1,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    29,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    -1,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    -1,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    -1,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,    -1,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    61,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    -1,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    -1,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    -1,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    -1,    54,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    86,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,   103,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,     1,    -1,
       3,     4,     5,     6,     7,     8,    -1,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    -1,    30,    31,    -1,
      33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    47,    48,    49,    -1,    51,    52,
      -1,    54,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,     3,     4,     5,     6,     7,     8,    -1,    82,
      -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,
     103,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    44,    -1,    46,    -1,    48,    -1,    -1,
      51,    52,    -1,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,     3,     4,     5,     6,     7,     8,
      -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,
      -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,   110,
      -1,   112,   113,   114,   115,    44,    -1,    46,    -1,    48,
      -1,    -1,    51,    52,    -1,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,     3,     4,     5,     6,
       7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,
      27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,
     109,   110,    -1,   112,   113,   114,   115,    44,    -1,    46,
      -1,    48,    -1,    -1,    51,    52,    -1,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,
     107,   108,   109,   110,    -1,   112,   113,   114,   115,    44,
      -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    26,    27,    -1,    -1,    -1,    -1,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,
      -1,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,
      -1,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    44,    -1,    46,    -1,    48,    -1,    -1,
      51,    52,    -1,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,
      -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,   110,
      -1,   112,   113,   114,   115,    44,    -1,    46,    -1,    48,
      -1,    -1,    51,    52,    -1,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,
      27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,
     109,   110,    -1,   112,   113,   114,   115,    44,    -1,    46,
      -1,    48,    -1,    -1,    51,    52,    -1,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,
     107,   108,   109,   110,    -1,   112,   113,   114,   115,    44,
      -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    26,    27,    -1,    -1,    -1,    -1,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,
      -1,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,
      -1,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    44,    -1,    46,    -1,    48,    -1,    -1,
      51,    52,    -1,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,
      -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,   110,
      -1,   112,   113,   114,   115,    44,    -1,    46,    -1,    48,
      -1,    -1,    51,    52,    -1,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,
      27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,
     109,   110,    -1,   112,   113,   114,   115,    44,    -1,    46,
      -1,    48,    -1,    -1,    51,    52,    -1,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,
     107,   108,   109,   110,    -1,   112,   113,   114,   115,    44,
      -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    26,    27,    -1,    -1,    -1,    -1,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,
      -1,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,
      -1,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    44,    -1,    46,    -1,    48,    -1,    -1,
      51,    52,    -1,    -1,    -1,    56,    -1,    58,    -1,    60,
       1,    -1,     3,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,    -1,
      -1,    -1,    43,    -1,   105,   106,   107,   108,   109,   110,
      -1,   112,   113,   114,   115,    56,    -1,    58,    59,    -1,
      -1,    -1,     1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    78,    79,    80,
      81,    -1,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,    43,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   113,   114,    -1,    -1,    -1,    56,    -1,    58,
      59,    -1,    -1,    -1,    -1,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,     3,    -1,    -1,
      79,    80,    81,    -1,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    -1,    -1,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   113,   114,    -1,    43,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      56,    -1,    58,    59,    -1,    -1,    -1,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
       3,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   113,   114,    -1,
      43,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     3,    -1,    56,    -1,    58,    59,    -1,    -1,    -1,
      -1,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,     3,    -1,    56,    -1,    58,    59,    -1,    -1,
     113,   114,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    78,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,     3,    -1,    56,    -1,    58,    59,    -1,
      -1,   113,   114,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    -1,    79,    80,
      81,    -1,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,     3,    -1,    56,    -1,    58,    59,
      -1,    -1,   113,   114,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,     3,    -1,    56,    -1,    58,
      59,    -1,    -1,   113,   114,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    -1,
      79,    80,    81,    -1,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    -1,    -1,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,     3,    -1,    56,    -1,
      58,    59,    -1,    -1,   113,   114,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,     3,    -1,    56,
      -1,    58,    59,    -1,    -1,   113,   114,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      -1,    -1,    79,    80,    81,    -1,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    -1,    -1,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,     3,    -1,
      56,    -1,    58,    59,    -1,    -1,   113,   114,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,     3,
      -1,    56,    -1,    58,    59,    -1,    -1,   113,   114,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
      -1,    -1,    56,    -1,    58,    59,    -1,    -1,   113,   114,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,     4,     5,     6,     7,     8,    -1,    -1,    -1,   113,
     114,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      43,    44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,
      -1,    -1,    -1,    56,    57,    58,    -1,    60,    -1,    62,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    26,    27,    -1,    -1,
      -1,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    43,    44,    -1,    46,    -1,    48,    -1,
      -1,    51,    52,    -1,    -1,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,
      -1,   101,    -1,    -1,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,    44,    -1,    46,    -1,
      48,    -1,    -1,    51,    52,    -1,    -1,    -1,    56,    57,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,    44,    -1,
      46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,    -1,
      56,    57,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,
      -1,    97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,
     106,   107,   108,   109,   110,    -1,   112,   113,   114,   115,
      44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,
      -1,    55,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    82,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,
      -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,    -1,
      -1,   105,   106,   107,   108,   109,   110,    -1,   112,   113,
     114,   115,    44,    -1,    46,    -1,    48,    -1,    -1,    51,
      52,    -1,    -1,    -1,    56,    57,    58,    -1,    60,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,
      -1,    -1,    -1,   105,   106,   107,   108,   109,   110,    -1,
     112,   113,   114,   115,    44,    -1,    46,    -1,    48,    -1,
      -1,    51,    52,    -1,    -1,    55,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,
      -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,    44,    -1,    46,    -1,
      48,    -1,    -1,    51,    52,    -1,    -1,    -1,    56,    -1,
      58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    82,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,
      -1,    -1,    -1,    -1,    -1,    43,    -1,   105,   106,   107,
     108,   109,   110,    -1,   112,   113,   114,   115,    56,    -1,
      58,    59,    -1,    -1,    62,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,    43,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   113,   114,    -1,    -1,    -1,
      56,    57,    58,    59,    -1,    -1,    -1,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,    43,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   113,   114,    -1,
      -1,    -1,    56,    57,    58,    59,    -1,    -1,    -1,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,    -1,    55,    56,    -1,    58,    59,    -1,    -1,   113,
     114,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    -1,    -1,    56,    57,    58,    59,    -1,    -1,
     113,   114,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,    -1,    -1,    56,    -1,    58,    59,    -1,
      -1,   113,   114,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    77,    -1,    79,    80,
      81,    -1,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,    -1,    -1,    -1,    -1,    56,    -1,
      58,    59,   113,   114,    62,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,    -1,    -1,    56,
      57,    58,    59,    -1,    -1,   113,   114,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      -1,    -1,    79,    80,    81,    -1,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    -1,    -1,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,    -1,    -1,
      56,    57,    58,    59,    -1,    -1,   113,   114,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,    -1,
      -1,    56,    -1,    58,    59,    -1,    -1,   113,   114,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
      56,    -1,    58,    59,    -1,    -1,    -1,    -1,   113,   114,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    75,
      -1,    -1,    78,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,    56,
      -1,    58,    59,    -1,    -1,    -1,    -1,   113,   114,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    79,    80,    81,    -1,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    -1,    -1,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,    56,    -1,
      58,    59,    -1,    -1,    -1,    -1,   113,   114,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,    56,    -1,    58,
      59,    -1,    -1,    -1,    -1,   113,   114,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    -1,    -1,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,    56,    -1,    58,    59,
      -1,    -1,    -1,    -1,   113,   114,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   113,   114
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   117,   118,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    30,    31,    33,    34,    35,    37,
      38,    39,    40,    41,    42,    44,    46,    47,    48,    49,
      50,    51,    52,    54,    56,    58,    60,    82,    86,    92,
      93,    97,   103,   105,   106,   107,   108,   109,   110,   112,
     113,   114,   115,   119,   120,   122,   123,   124,   126,   127,
     129,   130,   131,   134,   136,   137,   145,   146,   147,   155,
     156,   157,   164,   166,   179,   181,   190,   192,   199,   200,
     201,   203,   205,   213,   214,   215,   217,   218,   220,   223,
     239,   244,   249,   253,   254,   256,   257,   259,   260,   261,
     263,   265,   269,   271,   272,   273,   274,   275,     3,    43,
      63,     3,     1,     6,   125,   256,     1,     6,    44,   259,
       1,     3,     1,     3,    14,     1,   259,     1,   256,   278,
       1,   259,     1,     1,   259,     1,     3,    43,     1,   259,
       1,     6,     1,     6,     1,     3,   259,   250,     1,     6,
       7,     1,   259,   261,     1,     6,     1,     3,     6,   216,
       1,     6,    32,   219,     1,     6,   221,   222,     1,     6,
     264,   270,     1,   259,    57,   259,   276,     1,     3,    43,
       6,   259,    43,    57,    62,   259,   275,   279,   266,   259,
       1,     3,   259,   275,   259,   259,   259,     1,     3,   275,
     259,   259,   259,   259,   259,   259,     4,     6,    26,    59,
     259,   259,     4,   256,   128,    31,    33,    44,   123,   135,
     123,   165,   180,   191,    45,   208,   211,   212,   123,     1,
      56,     3,    56,    58,    59,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    79,    80,    81,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     113,   114,   260,    75,    78,     1,     4,     5,     7,     8,
      51,    52,    97,   121,   255,   259,     3,     3,    78,    75,
       3,    43,     3,    43,     3,     3,     3,     3,    43,     3,
      43,     3,    75,    78,    91,     3,     3,     3,     3,     3,
       3,   123,     3,     3,     3,   224,     3,   245,     3,     3,
       1,     6,   251,   252,     3,     3,     3,     3,     3,     3,
      75,     3,     3,    78,     3,     1,     6,     7,     1,     3,
      32,    78,     3,    75,     3,    78,     3,     1,    56,   267,
     267,     3,     3,     1,    57,    78,   277,     3,   132,   123,
     240,    55,    57,   259,    57,    43,    62,     1,    57,     1,
      57,    78,     1,     6,   206,   207,   268,     3,     3,     3,
       3,     4,     6,    26,   144,   144,   167,   182,   144,     1,
       3,    43,   144,   209,   210,     3,     1,   206,    55,   275,
     101,   259,     6,   259,   259,   259,   259,   259,   259,   259,
     259,   259,   259,   259,   259,     1,   259,   259,   259,   259,
     259,   259,   259,   259,   259,     6,   259,   259,     3,   258,
     258,   258,   258,   258,   258,   258,   258,   258,   258,   258,
     259,   259,     3,     4,     3,   256,   259,   151,   259,   256,
       1,   259,     1,    56,   225,   226,     1,    32,   228,   246,
       3,    78,     1,     1,   255,     6,     3,     3,    76,     3,
      76,     3,     6,     7,     6,     6,     7,   121,   222,     3,
     206,   208,   208,   259,   144,     3,    57,    57,   259,   259,
      57,   259,    62,     1,    62,    78,   208,     9,   123,    16,
      17,   138,   140,   141,   143,     1,     3,    23,    24,   168,
     173,   175,     1,     3,    23,   173,   183,    29,   193,   194,
     195,   196,     3,    43,     9,   144,   123,   204,     1,    55,
       1,    55,   259,    57,    78,     1,    43,   259,   259,   259,
     259,   259,   259,   259,   259,   259,   259,   259,     3,    78,
      75,   149,    77,     3,   148,     3,   206,   232,   228,     3,
       6,   229,   230,     3,   247,     1,   252,     3,     3,     6,
       6,     3,    76,    91,     3,    76,    91,     1,    55,   144,
     144,     9,   241,    43,    57,    62,   207,   144,     3,     1,
       3,     1,   259,     9,   139,   142,     3,     1,     6,     7,
       8,   121,   177,   178,     1,     9,   174,     3,     1,     4,
       6,   188,   189,     9,     1,     3,     4,     6,    91,   197,
     198,     9,   195,   144,     3,     9,    55,   202,     3,    43,
     262,    57,   275,     1,   259,   275,   259,     3,    43,   150,
       1,   259,   150,     1,    55,     3,     6,     9,    36,    44,
      45,    53,   200,   218,   233,   234,   236,   237,     3,    56,
     231,    78,     3,     9,   200,   218,   234,   236,   248,     3,
       3,     6,     6,     6,     6,     3,     9,     9,   133,   259,
       3,     6,     9,   218,   242,   259,   259,    61,     3,     3,
       3,     3,   144,   144,     3,    43,    77,     3,    43,    78,
       3,     3,    43,   176,     3,    43,     3,    43,    78,     3,
       3,   256,     3,    78,    91,     3,     3,    43,    55,    55,
       3,    19,    20,    21,   123,   153,   154,   158,   160,   162,
     123,     1,    78,   152,   227,    75,     3,     3,     6,     6,
      55,   275,   230,     3,     3,     3,     3,     3,     3,     3,
      75,    78,   243,     3,    57,   138,   171,   172,   121,   169,
     170,   178,   144,   123,   186,   187,   184,   185,   189,     3,
     198,   256,     3,     1,     3,    43,     1,     3,    43,     1,
       3,    43,     9,   153,     1,   259,    55,   259,   235,    75,
       3,    55,   255,   144,   123,   144,   123,   144,   123,   144,
     123,     3,     3,   159,   123,     3,   161,   123,     3,   163,
     123,     3,     3,   208,   255,   201,   205,   238,   243,   144,
     144,   144,   144,     3,    56,     9,   201,     9,     9,     9,
       9,   206,     3,     3,     3,     3,     3,    55
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
#line 207 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); }
    break;

  case 7:
#line 208 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); }
    break;

  case 8:
#line 212 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat)=0; }
    break;

  case 10:
#line 215 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 11:
#line 220 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 12:
#line 225 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 13:
#line 230 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 14:
#line 235 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 19:
#line 246 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); }
    break;

  case 20:
#line 251 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), false );
      }
    break;

  case 21:
#line 257 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), true );
      }
    break;

  case 22:
#line 263 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
      }
    break;

  case 23:
#line 269 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); }
    break;

  case 24:
#line 270 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_stat)=0;}
    break;

  case 25:
#line 271 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; }
    break;

  case 26:
#line 272 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; }
    break;

  case 27:
#line 273 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; }
    break;

  case 28:
#line 274 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; }
    break;

  case 29:
#line 275 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;}
    break;

  case 30:
#line 280 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (! COMPILER->isInteractive()) && ((!(yyvsp[(1) - (2)].fal_val)->isExpr()) || (!(yyvsp[(1) - (2)].fal_val)->asExpr()->isStandAlone()) ) )
         {
            Falcon::StmtFunction *func = COMPILER->getFunctionContext();
            if ( func == 0 || ! func->isLambda() )
               COMPILER->raiseError(Falcon::e_noeffect );
         }

         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) );
      }
    break;

  case 31:
#line 291 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      }
    break;

  case 32:
#line 297 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 50:
#line 331 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_def );
   }
    break;

  case 51:
#line 334 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   }
    break;

  case 52:
#line 340 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   }
    break;

  case 53:
#line 349 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; }
    break;

  case 54:
#line 351 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); }
    break;

  case 55:
#line 355 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 56:
#line 362 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 57:
#line 369 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 58:
#line 377 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 59:
#line 378 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; }
    break;

  case 60:
#line 382 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 61:
#line 383 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 62:
#line 387 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 63:
#line 394 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = static_cast<Falcon::StmtLoop* >(COMPILER->getContext());
         w->setCondition((yyvsp[(6) - (7)].fal_val));
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 64:
#line 402 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 65:
#line 408 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_loop );
      (yyval.fal_stat) = 0;
   }
    break;

  case 66:
#line 415 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 67:
#line 416 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(1) - (1)].fal_val); }
    break;

  case 68:
#line 420 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
    break;

  case 69:
#line 428 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 70:
#line 435 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      }
    break;

  case 71:
#line 445 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 72:
#line 446 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; }
    break;

  case 73:
#line 450 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 74:
#line 451 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 77:
#line 458 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
    break;

  case 80:
#line 468 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); }
    break;

  case 81:
#line 473 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
    break;

  case 83:
#line 485 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 84:
#line 486 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; }
    break;

  case 86:
#line 491 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   }
    break;

  case 87:
#line 498 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      }
    break;

  case 88:
#line 507 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      }
    break;

  case 89:
#line 515 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      }
    break;

  case 90:
#line 525 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      }
    break;

  case 91:
#line 534 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      }
    break;

  case 92:
#line 543 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (4)].fal_adecl);
         f = new Falcon::StmtForin( LINE, decl, (yyvsp[(4) - (4)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 93:
#line 554 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 94:
#line 563 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) );
         Falcon::ArrayDecl *decl = new Falcon::ArrayDecl();
         decl->pushBack( (yyvsp[(2) - (4)].fal_val) );
         f = new Falcon::StmtForin( LINE, decl, (yyvsp[(4) - (4)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 95:
#line 576 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 96:
#line 586 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 97:
#line 591 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 98:
#line 599 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 100:
#line 612 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::RangeDecl* rd = new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val),
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(3) - (4)].fal_val))), (yyvsp[(4) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( rd );
      }
    break;

  case 101:
#line 618 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val), 0 ) );
      }
    break;

  case 102:
#line 622 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (3)].fal_val), 0, 0 ) );
      }
    break;

  case 103:
#line 628 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 104:
#line 629 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 105:
#line 630 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 108:
#line 639 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      }
    break;

  case 112:
#line 653 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 113:
#line 666 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      }
    break;

  case 114:
#line 674 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 115:
#line 678 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 116:
#line 684 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 117:
#line 690 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      }
    break;

  case 118:
#line 697 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 119:
#line 702 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 120:
#line 711 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   }
    break;

  case 121:
#line 720 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 122:
#line 732 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 123:
#line 734 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 124:
#line 743 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
    break;

  case 125:
#line 747 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 126:
#line 759 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 127:
#line 760 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 128:
#line 769 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); }
    break;

  case 129:
#line 773 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 130:
#line 787 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 131:
#line 789 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 132:
#line 798 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
    break;

  case 133:
#line 802 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 134:
#line 810 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 135:
#line 819 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 136:
#line 821 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 139:
#line 830 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); }
    break;

  case 141:
#line 836 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 143:
#line 846 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 144:
#line 854 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 145:
#line 858 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 147:
#line 870 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 148:
#line 880 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 150:
#line 889 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 154:
#line 903 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); }
    break;

  case 156:
#line 907 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 159:
#line 919 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }
    break;

  case 160:
#line 928 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 161:
#line 940 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 162:
#line 951 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 163:
#line 962 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 164:
#line 982 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 165:
#line 990 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 166:
#line 999 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 167:
#line 1001 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 170:
#line 1010 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); }
    break;

  case 172:
#line 1016 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 174:
#line 1026 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 175:
#line 1035 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 176:
#line 1039 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 178:
#line 1051 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 179:
#line 1061 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 183:
#line 1075 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 184:
#line 1087 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 185:
#line 1107 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   }
    break;

  case 186:
#line 1114 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }
    break;

  case 187:
#line 1124 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
    break;

  case 189:
#line 1133 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); }
    break;

  case 195:
#line 1153 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 196:
#line 1171 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 197:
#line 1191 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 198:
#line 1201 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 199:
#line 1212 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }
    break;

  case 202:
#line 1225 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 203:
#line 1237 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 204:
#line 1259 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 205:
#line 1260 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; }
    break;

  case 206:
#line 1272 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 207:
#line 1278 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 209:
#line 1287 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 210:
#line 1288 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 211:
#line 1291 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
    break;

  case 213:
#line 1296 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 214:
#line 1297 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 215:
#line 1304 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
                // is this a setter/getter?
                if( ( (yyvsp[(2) - (2)].stringp)->find( "__set_" ) == 0 || (yyvsp[(2) - (2)].stringp)->find( "__get_" ) == 0 ) && (yyvsp[(2) - (2)].stringp)->length() > 6 )
                {
                   Falcon::String *pname = COMPILER->addString( (yyvsp[(2) - (2)].stringp)->subString( 6 ));
                   Falcon::VarDef *pd = cd->getProperty( *pname );
                   if( pd == 0 )
                   {
                     pd = new Falcon::VarDef;
                     cd->addProperty( pname, pd );
                     pd->setReflective( Falcon::e_reflectSetGet, 0xFFFFFFFF );
                   }
                   else if( ! pd->isReflective() )
                   {
                     COMPILER->raiseError(Falcon::e_prop_adef, *pname );
                   }

                }
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

  case 219:
#line 1382 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 221:
#line 1399 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 222:
#line 1405 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 223:
#line 1410 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 224:
#line 1416 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 226:
#line 1425 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); }
    break;

  case 228:
#line 1430 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
    break;

  case 229:
#line 1440 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 230:
#line 1443 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; }
    break;

  case 231:
#line 1452 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 232:
#line 1462 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      }
    break;

  case 233:
#line 1467 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      }
    break;

  case 234:
#line 1479 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 235:
#line 1488 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      }
    break;

  case 236:
#line 1495 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      }
    break;

  case 237:
#line 1503 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      }
    break;

  case 238:
#line 1508 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      }
    break;

  case 239:
#line 1516 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (3)].fal_genericList) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 240:
#line 1521 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 241:
#line 1526 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 242:
#line 1531 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 243:
#line 1551 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 244:
#line 1570 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 245:
#line 1575 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 246:
#line 1580 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 247:
#line 1585 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 248:
#line 1599 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 249:
#line 1604 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 250:
#line 1609 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 251:
#line 1614 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 252:
#line 1619 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 253:
#line 1628 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addAttribute( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val), LINE );
     }
    break;

  case 254:
#line 1633 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_attrdecl );
     }
    break;

  case 255:
#line 1640 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *(yyvsp[(1) - (1)].stringp) ) );
         (yyval.fal_genericList) = lst;
      }
    break;

  case 256:
#line 1646 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(1) - (3)].fal_genericList)->pushBack( new Falcon::String( *(yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_genericList) = (yyvsp[(1) - (3)].fal_genericList);
      }
    break;

  case 257:
#line 1658 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 258:
#line 1663 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 261:
#line 1676 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 262:
#line 1680 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 263:
#line 1684 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 264:
#line 1697 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 265:
#line 1729 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls != 0 )
         {
            if ( cls->ctorFunction() == 0  )
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
      }
    break;

  case 267:
#line 1763 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 270:
#line 1771 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 271:
#line 1772 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 276:
#line 1789 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol((yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            cls->addInitExpression(
               new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_inherit,
                     new Falcon::Value( sym ), (yyvsp[(2) - (2)].fal_val) ) ) );
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
            delete (yyvsp[(2) - (2)].fal_val);
         }
      }
    break;

  case 277:
#line 1812 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 278:
#line 1813 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 279:
#line 1815 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = (yyvsp[(2) - (3)].fal_adecl) == 0 ? 0 : new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
   }
    break;

  case 283:
#line 1828 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 284:
#line 1831 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 288:
#line 1855 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
            COMPILER->pushFunctionContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      }
    break;

  case 289:
#line 1880 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
         COMPILER->popFunctionContext();
      }
    break;

  case 290:
#line 1890 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 291:
#line 1912 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 292:
#line 1941 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_stat)=0;}
    break;

  case 295:
#line 1955 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 296:
#line 1989 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 300:
#line 2006 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 302:
#line 2011 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 305:
#line 2026 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 306:
#line 2066 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

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

  case 308:
#line 2094 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 312:
#line 2106 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 313:
#line 2109 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 316:
#line 2138 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 317:
#line 2143 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 319:
#line 2157 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 320:
#line 2162 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 322:
#line 2168 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 323:
#line 2175 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 324:
#line 2190 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 325:
#line 2191 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 326:
#line 2192 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 327:
#line 2200 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 328:
#line 2201 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 329:
#line 2202 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 330:
#line 2203 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 331:
#line 2204 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 332:
#line 2205 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 333:
#line 2209 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 334:
#line 2210 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 335:
#line 2211 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 336:
#line 2212 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 337:
#line 2213 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 338:
#line 2214 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 339:
#line 2219 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 341:
#line 2237 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 342:
#line 2238 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtFunction *sfunc = COMPILER->getFunctionContext();
      if ( sfunc == 0 ) {
         COMPILER->raiseError(Falcon::e_fself_outside, COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value();
      }
      else
      {
         (yyval.fal_val) = new Falcon::Value( sfunc->symbol() );
      }
   }
    break;

  case 347:
#line 2266 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( (yyvsp[(2) - (2)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 348:
#line 2267 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, "%d", (int)(yyvsp[(2) - (2)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 349:
#line 2268 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString("self") ); /* do not add the symbol to the compiler */ }
    break;

  case 350:
#line 2269 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyvsp[(3) - (3)].stringp)->prepend( "." ); (yyval.fal_val)->setLBind( (yyvsp[(3) - (3)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 351:
#line 2270 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, ".%d", (int)(yyvsp[(3) - (3)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 352:
#line 2271 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(".self") ); /* do not add the symbol to the compiler */ }
    break;

  case 353:
#line 2272 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 354:
#line 2273 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_fbind, new Falcon::Value((yyvsp[(1) - (3)].stringp)), (yyvsp[(3) - (3)].fal_val)) ); }
    break;

  case 355:
#line 2274 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            // is this an immediate string sum ?
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isString() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str += *(yyvsp[(4) - (4)].fal_val)->asString();
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.writeNumber( (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
         }
    break;

  case 356:
#line 2300 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 357:
#line 2301 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( (yyvsp[(1) - (4)].fal_val)->asString()->length() );
                  for( int i = 0; i < (yyvsp[(4) - (4)].fal_val)->asInteger(); ++i )
                  {
                     str.append( *(yyvsp[(1) - (4)].fal_val)->asString()  );
                  }
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 358:
#line 2321 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if( (yyvsp[(1) - (4)].fal_val)->asString()->length() == 0 )
               {
                  COMPILER->raiseError( Falcon::e_invop );
               }
               else {

                  if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
                  {
                     Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                     str.setCharAt( str.length()-1, str.getCharAt(str.length()-1) + (yyvsp[(4) - (4)].fal_val)->asInteger() );
                     (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                     delete (yyvsp[(4) - (4)].fal_val);
                     (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
                  }
                  else
                     (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
               }
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 359:
#line 2345 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.append( (Falcon::uint32) (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 360:
#line 2362 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 361:
#line 2363 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 362:
#line 2364 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 363:
#line 2365 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 364:
#line 2366 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 365:
#line 2367 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 366:
#line 2368 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 367:
#line 2369 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 368:
#line 2370 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 369:
#line 2371 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 370:
#line 2372 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 371:
#line 2373 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 372:
#line 2374 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 373:
#line 2375 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 374:
#line 2376 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 375:
#line 2377 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 376:
#line 2378 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 377:
#line 2379 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 378:
#line 2380 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 379:
#line 2381 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 380:
#line 2382 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 381:
#line 2383 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 382:
#line 2384 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 383:
#line 2385 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 384:
#line 2386 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 385:
#line 2387 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 386:
#line 2388 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 387:
#line 2389 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eval, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 388:
#line 2390 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 389:
#line 2391 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_deoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 390:
#line 2392 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_isoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 391:
#line 2393 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_xoroob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 398:
#line 2401 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 399:
#line 2406 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 400:
#line 2410 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 401:
#line 2415 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 402:
#line 2421 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 405:
#line 2433 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 406:
#line 2438 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 407:
#line 2445 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 408:
#line 2446 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 409:
#line 2447 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 410:
#line 2448 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 411:
#line 2449 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 412:
#line 2450 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 413:
#line 2451 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 414:
#line 2452 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 415:
#line 2453 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 416:
#line 2454 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 417:
#line 2455 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 418:
#line 2456 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 419:
#line 2461 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 420:
#line 2464 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 421:
#line 2467 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 422:
#line 2470 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 423:
#line 2473 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 424:
#line 2480 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 425:
#line 2486 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 426:
#line 2490 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 427:
#line 2491 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 428:
#line 2500 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 429:
#line 2535 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 430:
#line 2543 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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

  case 431:
#line 2577 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StatementList *stmt = COMPILER->getContextSet();
         if( stmt->size() == 1 && stmt->back()->type() == Falcon::Statement::t_autoexp )
         {
            // wrap it around a return, so A is not nilled.
            Falcon::StmtAutoexpr *ae = static_cast<Falcon::StmtAutoexpr *>( stmt->pop_back() );
            stmt->push_back( new Falcon::StmtReturn( 1, ae->value()->clone() ) );

            // we don't need the expression anymore.
            delete ae;
         }

         (yyval.fal_val) = COMPILER->closeClosure();
      }
    break;

  case 433:
#line 2596 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 434:
#line 2600 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 436:
#line 2608 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 437:
#line 2612 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 438:
#line 2619 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
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
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 439:
#line 2653 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 440:
#line 2669 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 441:
#line 2674 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 442:
#line 2681 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 443:
#line 2688 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 444:
#line 2697 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 445:
#line 2699 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 446:
#line 2703 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 447:
#line 2710 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 448:
#line 2712 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 449:
#line 2716 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 450:
#line 2724 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 451:
#line 2725 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 452:
#line 2727 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 453:
#line 2734 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 454:
#line 2735 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 455:
#line 2739 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 456:
#line 2740 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 459:
#line 2747 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 460:
#line 2753 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 461:
#line 2760 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 462:
#line 2761 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;


/* Line 1267 of yacc.c.  */
#line 6716 "/home/user/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2765 "/home/user/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


